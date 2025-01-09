#pragma once

#include <chowdsp_plugin_base/chowdsp_plugin_base.h>
#include "lstm_model.h"

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

private:
    LSTM_Model lstm_model {};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (Neural_Pruning_Plugin)
};
