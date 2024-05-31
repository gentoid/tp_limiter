use nih_plug::{prelude::*, util::StftHelper};
use nih_plug_iced::IcedState;
use realfft::{
    num_complex::{Complex32, ComplexFloat},
    ComplexToReal, RealFftPlanner, RealToComplex,
};
use std::sync::{atomic, Arc};

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
}

#[derive(Default)]
struct Values {
    min: AtomicF32,
    max: AtomicF32,
    abs: AtomicF32,
}

#[derive(Params)]
struct TpLimiterParams {
    #[persist = "gui-state"]
    gui_state: Arc<IcedState>,

    #[id = "gain"]
    pub gain: FloatParam,
    #[id = "release"]
    pub release_ms: FloatParam,
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
        Self {
            params: Arc::new(TpLimiterParams::default()),

            // We'll process the input in `WINDOW_SIZE` chunks, but our FFT window is slightly
            // larger to account for time domain aliasing so we'll need to add some padding ot each
            // block.
            stft: StftHelper::new(2, WINDOW_SIZE, FFT_WINDOW_SIZE - WINDOW_SIZE),

            r2c_plan,
            c2r_plan,
            complex_fft_buffer,

            values: Arc::new(Values::default()),
        }
    }
}

impl Default for TpLimiterParams {
    fn default() -> Self {
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
                200.0,
                FloatRange::Skewed {
                    min: 0.1,
                    max: 10000.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_step_size(0.1)
            .with_unit(" ms"),
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
        _buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        // The plugin's latency consists of the block size from the overlap-add procedure and half
        // of the filter kernel's size (since we're using a linear phase/symmetrical convolution
        // kernel)
        context.set_latency_samples(self.stft.latency_samples() + (FILTER_WINDOW_SIZE as u32 / 2));
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
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        self.stft
            .process_overlap_add(buffer, 1, |_channel_idx, real_fft_buffer| {
                // Forward FFT, `real_fft_buffer` already is already padded with zeroes, and the
                // padding from the last iteration will have already been added back to the start of
                // the buffer
                self.r2c_plan
                    .process_with_scratch(real_fft_buffer, &mut self.complex_fft_buffer, &mut [])
                    .unwrap();

                let mut min: f32 = 0.0;
                let mut max: f32 = 0.0;
                let mut abs: f32 = 0.0;
                // As per the convolution theorem we can simply multiply these two buffers. We'll
                // also apply the gain compensation at this point.
                for fft_bin in self.complex_fft_buffer.iter_mut() {
                    *fft_bin *= GAIN_COMPENSATION;
                    min = min.min(fft_bin.re);
                    max = max.max(fft_bin.re);
                    abs = abs.max(fft_bin.abs());
                }

                self.values.min.store(
                    self.values.min.load(atomic::Ordering::Relaxed).min(min),
                    atomic::Ordering::Relaxed,
                );
                self.values.max.store(
                    self.values.max.load(atomic::Ordering::Relaxed).max(max),
                    atomic::Ordering::Relaxed,
                );
                self.values.abs.store(
                    self.values.abs.load(atomic::Ordering::Relaxed).max(abs),
                    atomic::Ordering::Relaxed,
                );

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
