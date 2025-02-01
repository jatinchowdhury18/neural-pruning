#include "plugin_editor.h"

struct Console : juce::Component
{
    juce::TextEditor console_log {};
    juce::TextButton clear_logs_button { "CLEAR LOGS" };
    Console_Logger& logger;

    explicit Console (Console_Logger& console_logger)
        : logger { console_logger }
    {
        console_log.setMultiLine (true);
        console_log.setFont (juce::FontOptions { "JetBrains Mono", 16.0f, juce::Font::plain });
        console_log.setReadOnly (true);
        logger.set_console (&console_log);
        addAndMakeVisible (console_log);

        clear_logs_button.onClick = [this]
        {
            logger.log_text.clear();
            logger.update_console();
        };
        addAndMakeVisible (clear_logs_button);
    }

    ~Console() override
    {
        logger.set_console (nullptr);
    }

    void resized() override
    {
        auto b = getLocalBounds();
        console_log.setBounds (b);
        clear_logs_button.setBounds (b.removeFromBottom (30).removeFromRight (50));
    }
};

struct Pruning_View : juce::Component
{
    juce::TextButton prune_find_button { "Find Pruning Candidate" };
    juce::Slider threshold_slider {};
    juce::TextButton prune_to_thresh_button { "Prune to Threshold" };
    juce::TextButton prune_button { "Prune" };
    juce::TextButton reset_button { "Reset" };
    juce::Label pruning_status_label {};

    chowdsp::ScopedCallbackList callbacks {};

    explicit Pruning_View (LSTM_Model& model, chowdsp::ComponentArena<>& arena)
    {
        addAndMakeVisible (prune_find_button);
        addAndMakeVisible (threshold_slider);
        addAndMakeVisible (prune_to_thresh_button);
        addAndMakeVisible (prune_button);
        prune_button.setEnabled (false);
        addAndMakeVisible (reset_button);

        prune_find_button.onClick = [this, &model]
        {
            prune_find_button.setEnabled (false);
            model.find_pruning_candidate();
        };
        reset_button.onClick = [&model] { model.reload_original_model(); };

        threshold_slider.setNormalisableRange (chowdsp::ParamUtils::createNormalisableRange (1.0e-3, 1.0e-1, 1.0e-2));
        threshold_slider.setSliderStyle (juce::Slider::SliderStyle::LinearHorizontal);
        threshold_slider.setColour (juce::Slider::ColourIds::textBoxBackgroundColourId, juce::Colours::black);

        prune_to_thresh_button.onClick = [this, &model]
        {
            prune_find_button.setEnabled (false);
            prune_to_thresh_button.setEnabled (false);
            prune_button.setEnabled (false);

            const auto thresh = static_cast<float> (threshold_slider.getValue());
            model.prune_model (thresh);
        };

        pruning_status_label.setJustificationType (juce::Justification::topLeft);
        pruning_status_label.setColour (juce::Label::textColourId, juce::Colours::black);
        set_status (chowdsp::format (arena.allocator, "Current hidden size: {}", model.current_hidden_size));
        addAndMakeVisible (pruning_status_label);

        callbacks += {
            model.new_pruning_candidate.connect (
                [this, &model, &arena] (int channel, float rms_error)
                {
                    prune_find_button.setEnabled (false);
                    prune_button.setEnabled (true);
                    set_status (chowdsp::format (arena.allocator,
                                                 "New pruning candidate: channel: {}, RMS error: {}",
                                                 channel,
                                                 rms_error));

                    prune_button.onClick = [this, channel, &model]
                    {
                        prune_button.setEnabled (false);
                        model.prune_channel (channel);
                    };
                }),
            model.model_changed.connect (
                [this, &model, &arena]
                {
                    prune_find_button.setEnabled (true);
                    prune_to_thresh_button.setEnabled (true);
                    prune_button.setEnabled (false);
                    set_status (chowdsp::format (arena.allocator, "Current hidden size: {}", model.current_hidden_size));
                }),
        };
    }

    void set_status (std::string_view status_text)
    {
        pruning_status_label.setText (chowdsp::toString (status_text), juce::sendNotification);
    }

    void resized() override
    {
        auto b = getLocalBounds();
        auto buttons_col = b; //.removeFromLeft (proportionOfWidth (0.2f));
        auto button_height = proportionOfHeight (0.1f);

        prune_find_button.setBounds (buttons_col.removeFromTop (button_height));

        auto thresh_area = buttons_col.removeFromTop (button_height);
        prune_to_thresh_button.setBounds (thresh_area.removeFromRight (proportionOfWidth (0.5f)));
        threshold_slider.setBounds (thresh_area);

        prune_button.setBounds (buttons_col.removeFromTop (button_height));
        reset_button.setBounds (buttons_col.removeFromTop (button_height));
        pruning_status_label.setBounds (buttons_col);
    }

    void paint (juce::Graphics& g) override
    {
        g.fillAll (juce::Colours::white);

        // static constexpr int draw_px = 2;
        // const auto matrix_dim = proportionOfWidth (0.01f);
        // const auto kernel_matrix_x = proportionOfWidth (0.25f);
        // const auto kernel_matrix_y = proportionOfHeight (0.05f);

        // const auto hidden_size = 24;
        // const auto matrix_rect = juce::Rectangle { hidden_size * matrix_dim, hidden_size * matrix_dim }
        //                             .withPosition (kernel_matrix_x, kernel_matrix_y);

        // g.setColour (juce::Colours::black);
        // g.drawRect (matrix_rect, draw_px);
    }
};

Plugin_Editor::Plugin_Editor (Neural_Pruning_Plugin& plugin)
    : AudioProcessorEditor { plugin }
{
    setSize (800, 600);
    auto b = getLocalBounds();

    auto* console = arena.allocate<Console> (plugin.logger);
    console->setBounds (b.removeFromBottom (250));
    addAndMakeVisible (console);

    auto* pruning_view = arena.allocate<Pruning_View> (plugin.lstm_model, arena);
    pruning_view->setBounds (b);
    addAndMakeVisible (pruning_view);

    chowdsp::log ("Creating plugin editor...");
}

Plugin_Editor::~Plugin_Editor()
{
    chowdsp::log ("Destroying plugin editor...");
}

void Plugin_Editor::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colours::black);
}
