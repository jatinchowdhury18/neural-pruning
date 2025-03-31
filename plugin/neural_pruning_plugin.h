#pragma once

#include <chowdsp_dsp_utils/chowdsp_dsp_utils.h>
#include <chowdsp_filters/chowdsp_filters.h>
#include <chowdsp_plugin_base/chowdsp_plugin_base.h>

#include "console_logger.h"
#include "lstm_model.h"

struct Params : chowdsp::ParamHolder
{
    chowdsp::FloatParameter::Ptr hidden_size {
        PID { "hidden_size", 100 },
        "Pruned Hidden Size",
        juce::NormalisableRange<float> { LSTM_Model::min_hidden_size,
                                         LSTM_Model::max_hidden_size,
                                         1 },
        LSTM_Model::max_hidden_size,
        &chowdsp::ParamUtils::floatValToStringDecimal<0>,
        &chowdsp::ParamUtils::stringToFloatVal,
    };

    chowdsp::EnumChoiceParameter<Ranking>::Ptr ranking {
        PID { "ranking", 100 },
        "Ranking",
        Ranking::Mean_Activations,
    };

    Params()
    {
        add (hidden_size, ranking);
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

    LSTM_Model lstm_model {};

    chowdsp::OnePoleSVF<float, chowdsp::OnePoleSVFType::Highpass> dc_blocker;

    using AAFilter = chowdsp::EllipticFilter<8>;
    chowdsp::Upsampler<float, AAFilter> upsampler;
    chowdsp::Downsampler<float, AAFilter, false> downsampler;

    chowdsp::ScopedCallbackList callbacks {};

private:
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (Neural_Pruning_Plugin)
};
