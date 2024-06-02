use std::sync::{Arc, atomic};

use nih_plug::{context::gui::GuiContext, editor::Editor};
use nih_plug_iced::*;

use crate::{TpLimiterParams, Values};

pub(crate) fn default_state() -> Arc<IcedState> {
    IcedState::from_size(300, 180)
}

pub(crate) fn create(
    editor_state: Arc<IcedState>,
    params: Arc<TpLimiterParams>,
    values: Arc<Values>,
) -> Option<Box<dyn Editor>> {
    create_iced_editor::<TpLimiterEditor>(editor_state, (params, values))
}

struct TpLimiterEditor {
    params: Arc<TpLimiterParams>,
    context: Arc<dyn GuiContext>,

    values: Arc<Values>,
    slider_state: widgets::param_slider::State,
}

#[derive(Debug, Clone, Copy)]
enum Message {
    ParamUpdate(widgets::ParamMessage),
}

impl IcedEditor for TpLimiterEditor {
    type Executor = executor::Default;

    type Message = Message;

    type InitializationFlags = (Arc<TpLimiterParams>, Arc<Values>);

    fn new(
        (params, values): Self::InitializationFlags,
        context: Arc<dyn GuiContext>,
    ) -> (Self, Command<Self::Message>) {
        let editor = TpLimiterEditor {
            params,
            context,
            values,
            slider_state: Default::default(),
        };

        (editor, Command::none())
    }

    fn context(&self) -> &dyn GuiContext {
        self.context.as_ref()
    }

    fn update(
        &mut self,
        _window: &mut WindowQueue,
        message: Self::Message,
    ) -> Command<Self::Message> {
        match message {
            Message::ParamUpdate(msg) => self.handle_param_message(msg),
        }

        Command::none()
    }

    fn view(&mut self) -> Element<'_, Self::Message> {
        Column::new()
            .align_items(Alignment::Center)
            .push(
                Text::new("TP Limiter")
                    .height(25.into())
                    .width(Length::Fill)
                    .horizontal_alignment(alignment::Horizontal::Center)
                    .vertical_alignment(alignment::Vertical::Center),
            )
            .push(
                widgets::ParamSlider::new(&mut self.slider_state, &self.params.gain)
                    .map(Message::ParamUpdate),
            )
            .push(
                Text::new(format!(
                    "ABS value: {:.3}",
                    self.values.abs.load(atomic::Ordering::Relaxed)
                ))
                .height(25.into())
                .width(Length::Fill)
                .horizontal_alignment(alignment::Horizontal::Center)
                .vertical_alignment(alignment::Vertical::Center),
            )
            .push(
                Text::new(format!("Release: {:.2} ms", self.params.release_ms.value()))
                    .height(25.into())
                    .width(Length::Fill)
                    .horizontal_alignment(alignment::Horizontal::Center)
                    .vertical_alignment(alignment::Vertical::Center),
            )
            .push(
                Text::new(format!(
                    "ABS value: {:07}",
                    self.values.blocks.load(atomic::Ordering::Relaxed)
                ))
                .height(25.into())
                .width(Length::Fill)
                .horizontal_alignment(alignment::Horizontal::Center)
                .vertical_alignment(alignment::Vertical::Center),
            )
            .into()
    }
}
