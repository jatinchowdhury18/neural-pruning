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

Plugin_Editor::Plugin_Editor (Neural_Pruning_Plugin& plugin)
    : AudioProcessorEditor { plugin }
{
    setSize (500, 400);
    auto b = getLocalBounds();

    auto* console = arena.allocate<Console> (plugin.logger);
    console->setBounds (b.removeFromBottom (250));
    addAndMakeVisible (console);

    auto* params_view = arena.allocate<chowdsp::ParametersView> (plugin.getState(), plugin.getState().params);
    params_view->setBounds (b);
    addAndMakeVisible (params_view);

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
