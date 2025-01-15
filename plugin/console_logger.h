#pragma once

#include <chowdsp_logging/chowdsp_logging.h>
#include <juce_gui_basics/juce_gui_basics.h>

struct Console_Logger : chowdsp::BaseLogger
{
    Console_Logger()
    {
        onLogMessage.connect ([this] (const juce::String& message)
        {
            log_text += message.toStdString() + "\n";
            update_console();
        });
        chowdsp::set_global_logger (this);
    }

    ~Console_Logger() override
    {
        chowdsp::set_global_logger (nullptr);
    }

    void set_console (juce::TextEditor* new_console)
    {
        console = new_console;
        update_console();
    }

    void update_console() const
    {
        jassert (juce::MessageManager::existsAndIsCurrentThread());
        if (console != nullptr)
        {
            console->setText (log_text, juce::sendNotification);
            console->moveCaretToEnd();
        }
    }

    std::string log_text {};
    juce::TextEditor* console { nullptr };
};
