use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use nih_plug::{prelude::*, util::StftHelper};
use nih_plug_iced::IcedState;
use realfft::{ComplexToReal, num_complex::Complex32, RealFftPlanner, RealToComplex};

mod gui;

/// The size of the windows we'll process at a time.
const WINDOW_SIZE: usize = 64;
/// The length of the filter's impulse response.
const FILTER_WINDOW_SIZE: usize = 33;
/// The length of the FFT window we will use to perform FFT convolution. This includes padding to
/// prevent time domain aliasing as a result of cyclic convolution.
const FFT_WINDOW_SIZE: usize = WINDOW_SIZE + FILTER_WINDOW_SIZE - 1;

/// The gain compensation we need to apply for the STFT process.
const GAIN_COMPENSATION: f32 = 1.0 / FFT_WINDOW_SIZE as f32;

struct TpLimiter {
    params: Arc<TpLimiterParams>,

    /// An adapter that performs most of the overlap-add algorithm for us.
    stft: StftHelper,

    /// The algorithm for the FFT operation.
    r2c_plan: Arc<dyn RealToComplex<f32>>,
    /// The algorithm for the IFFT operation.
    c2r_plan: Arc<dyn ComplexToReal<f32>>,
    /// The output of our real->complex FFT.
    complex_fft_buffer: Vec<Complex32>,

    values: Arc<Values>,
    envelope_follower: Smoother<f32>,
    release_changed: Arc<AtomicBool>,
}

#[derive(Default)]
struct Values {
    abs: AtomicF32,
}

#[derive(Enum, PartialEq)]
enum OversamplingOptions {
    X1,
    X4,
    X16,
}

#[derive(Params)]
struct TpLimiterParams {
    #[persist = "gui-state"]
    gui_state: Arc<IcedState>,

    #[id = "gain"]
    pub gain: FloatParam,
    #[id = "release"]
    pub release_ms: FloatParam,
    #[id = "oversampling"]
    pub oversampling_factor: EnumParam<OversamplingOptions>,
}

impl Default for TpLimiter {
    fn default() -> Self {
        let mut planner = RealFftPlanner::new();
        let r2c_plan = planner.plan_fft_forward(FFT_WINDOW_SIZE);
        let c2r_plan = planner.plan_fft_inverse(FFT_WINDOW_SIZE);
        let mut real_fft_buffer = r2c_plan.make_input_vec();
        let mut complex_fft_buffer = r2c_plan.make_output_vec();

        // Build a super simple low-pass filter from one of the built in window functions
        let mut filter_window = util::window::hann(FILTER_WINDOW_SIZE);
        // And make sure to normalize this so convolution sums to 1
        let filter_normalization_factor = filter_window.iter().sum::<f32>().recip();
        for sample in &mut filter_window {
            *sample *= filter_normalization_factor;
        }
        real_fft_buffer[0..FILTER_WINDOW_SIZE].copy_from_slice(&filter_window);

        // RustFFT doesn't actually need a scratch buffer here, so we'll pass an empty buffer
        // instead
        r2c_plan
            .process_with_scratch(&mut real_fft_buffer, &mut complex_fft_buffer, &mut [])
            .unwrap();

        let default_release: f32 = 200.0;
        let envelope_follower = Smoother::new(SmoothingStyle::Exponential(default_release));
        envelope_follower.reset(util::db_to_gain(0.0));
        envelope_follower.set_target(44100.0, util::db_to_gain(0.0));

        let release_changed: Arc<AtomicBool> = Arc::new(true.into());
        let release_changed_clone = release_changed.clone();
        let on_release_changed =
            Arc::new(move |_| release_changed_clone.store(true, Ordering::SeqCst));

        Self {
            params: Arc::new(TpLimiterParams::new(default_release, on_release_changed)),

            // We'll process the input in `WINDOW_SIZE` chunks, but our FFT window is slightly
            // larger to account for time domain aliasing, so we'll need to add some padding on each
            // block.
            stft: StftHelper::new(2, WINDOW_SIZE, FFT_WINDOW_SIZE - WINDOW_SIZE),

            r2c_plan,
            c2r_plan,
            complex_fft_buffer,

            values: Arc::new(Values { abs: 0.0.into() }),
            envelope_follower,
            release_changed,
        }
    }
}

impl TpLimiterParams {
    fn new(default_release: f32, on_changed: Arc<dyn Fn(f32) + Send + Sync>) -> Self {
        Self {
            gui_state: gui::default_state(),
            // This gain is stored as linear gain. NIH-plug comes with useful conversion functions
            // to treat these kinds of parameters as if we were dealing with decibels. Storing this
            // as decibels is easier to work with, but requires a conversion for every sample.
            gain: FloatParam::new(
                "Gain",
                util::db_to_gain(0.0),
                FloatRange::Skewed {
                    min: util::db_to_gain(-30.0),
                    max: util::db_to_gain(30.0),
                    // This makes the range appear as if it was linear when displaying the values as
                    // decibels
                    factor: FloatRange::gain_skew_factor(-30.0, 30.0),
                },
            )
            // Because the gain parameter is stored as linear gain instead of storing the value as
            // decibels, we need logarithmic smoothing
            .with_smoother(SmoothingStyle::Logarithmic(50.0))
            .with_unit(" dB")
            // There are many predefined formatters we can use here. If the gain was stored as
            // decibels instead of as a linear gain value, we could have also used the
            // `.with_step_size(0.1)` function to get internal rounding.
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),
            release_ms: FloatParam::new(
                "Release",
                default_release,
                FloatRange::Skewed {
                    min: 0.1,
                    max: 100000.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_step_size(0.1)
            .with_unit(" ms")
            .with_callback(on_changed),
            oversampling_factor: EnumParam::new("Oversampling", OversamplingOptions::X1),
        }
    }
}

impl Plugin for TpLimiter {
    const NAME: &'static str = "True Peak Limiter";
    const VENDOR: &'static str = "Viktor Lazarev";
    const URL: &'static str = env!("CARGO_PKG_HOMEPAGE");
    const EMAIL: &'static str = "taurus101v@gmail.com";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // The first audio IO layout is used as the default. The other layouts may be selected either
    // explicitly or automatically by the host or the user depending on the plugin API/backend.
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),

        aux_input_ports: &[],
        aux_output_ports: &[],

        // Individual ports and the layout as a whole can be named here. By default these names
        // are generated as needed. This layout will be called 'Stereo', while a layout with
        // only one input and output channel would be called 'Mono'.
        names: PortNames::const_default(),
    }];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    // If the plugin can send or receive SysEx messages, it can define a type to wrap around those
    // messages here. The type implements the `SysExMessage` trait, which allows conversion to and
    // from plain byte buffers.
    type SysExMessage = ();
    // More advanced plugins can use this to run expensive background tasks. See the field's
    // documentation for more information. `()` means that the plugin does not have any background
    // tasks.
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        gui::create(
            self.params.gui_state.clone(),
            self.params.clone(),
            self.values.clone(),
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        // The plugin's latency consists of the block size from the overlap-add procedure and half
        // of the filter kernel's size (since we're using a linear phase/symmetrical convolution
        // kernel)
        context.set_latency_samples(self.stft.latency_samples() + (FILTER_WINDOW_SIZE as u32 / 2));
        self.envelope_follower
            .set_target(buffer_config.sample_rate, util::db_to_gain(0.0));
        true
    }

    fn reset(&mut self) {
        // Reset buffers and envelopes here. This can be called from the audio thread and may not
        // allocate. You can remove this function if you do not need it.

        // Normally we'd also initialize the STFT helper for the correct channel count here, but we
        // only do stereo so that's not necessary. Setting the block size also zeroes out the
        // buffers.
        self.stft.set_block_size(WINDOW_SIZE);
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        if self.release_changed.load(Ordering::SeqCst) {
            self.release_changed.store(false, Ordering::SeqCst);
            let prev_value = self.envelope_follower.previous_value();

            self.envelope_follower =
                Smoother::new(SmoothingStyle::Exponential(self.params.release_ms.value()));

            self.envelope_follower.reset(prev_value);
            self.envelope_follower
                .set_target(context.transport().sample_rate, util::db_to_gain(0.0));
        }

        if buffer.samples() == 0 {
            return ProcessStatus::Normal;
        }

        let mut envelope_value: f32;

        for samples in buffer.as_slice_immutable() {
            for sample in samples.iter() {
                let abs = 1.0f32.max(sample.abs()); // util::db_to_gain(0.0)
                envelope_value = self.envelope_follower.next();

                if abs > envelope_value {
                    self.envelope_follower.reset(abs);
                    self.envelope_follower
                        .set_target(context.transport().sample_rate, util::db_to_gain(0.0));
                }
            }

            // NOTE: calculating a single channel only
            break;
        }

        self.values
            .abs
            .store(self.envelope_follower.previous_value(), Ordering::Relaxed);

        self.stft
            .process_overlap_add(buffer, 1, |_channel_idx, real_fft_buffer| {
                // Forward FFT, `real_fft_buffer` is already padded with zeroes, and the
                // padding from the last iteration will have already been added back to the start of
                // the buffer
                self.r2c_plan
                    .process_with_scratch(real_fft_buffer, &mut self.complex_fft_buffer, &mut [])
                    .unwrap();

                // TODO: where's oversampling going to be introduced?

                // As per the convolution theorem we can simply multiply these two buffers. We'll
                // also apply the gain compensation at this point.
                for fft_bin in self.complex_fft_buffer.iter_mut() {
                    *fft_bin *= GAIN_COMPENSATION;
                }

                // Inverse FFT back into the scratch buffer. This will be added to a ring buffer
                // which gets written back to the host at a one block delay.
                self.c2r_plan
                    .process_with_scratch(&mut self.complex_fft_buffer, real_fft_buffer, &mut [])
                    .unwrap();
            });

        ProcessStatus::Normal
    }
}

impl ClapPlugin for TpLimiter {
    const CLAP_ID: &'static str = "com.gentoid-dev.tp-limiter";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("One more True Peak Limiter (CLAP) plugin // WIP");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;

    // Don't forget to change these features
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Limiter,
    ];
}

nih_export_clap!(TpLimiter);
