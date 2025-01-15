#pragma once

#include <chowdsp_plugin_base/chowdsp_plugin_base.h>
#include <chowdsp_filters/chowdsp_filters.h>

#include "lstm_model.h"
#include "console_logger.h"

struct Params : chowdsp::ParamHolder
{
    chowdsp::PercentParameter::Ptr gain {
        PID { "gain", 100 },
        "Gain",
        0.5f,
    };

    Params()
    {
        add (gain);
    }
};

using State = chowdsp::PluginStateImpl<Params>;

class Neural_Pruning_Plugin : public chowdsp::PluginBase<State>
{
public:
    Neural_Pruning_Plugin();

    void prepareToPlay (double sample_rate, int samples_per_block) override;
    void releaseResources() override;
    void processAudioBlock (juce::AudioBuffer<float>& buffer) override;

    juce::AudioProcessorEditor* createEditor() override;

    Console_Logger logger {};

private:
    LSTM_Model lstm_model {};

    chowdsp::OnePoleSVF<float, chowdsp::OnePoleSVFType::Highpass> dc_blocker;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (Neural_Pruning_Plugin)
};
