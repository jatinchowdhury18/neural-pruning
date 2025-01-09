#pragma once

#include <chowdsp_plugin_base/chowdsp_plugin_base.h>
#include <RTNeural/RTNeural.h>

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

struct LSTM_Model
{
    static constexpr int input_size = 2;
    static constexpr int hidden_size = 24;

    using Model = RTNeural::ModelT<float,
                               input_size,
                               1,
                               RTNeural::LSTMLayerT<float, input_size, hidden_size>,
                               RTNeural::DenseT<float, hidden_size, 1>>;
    Model model {};

    void load_model (const nlohmann::json& model_json)
    {
        const auto& state_dict = model_json.at ("state_dict");
        RTNeural::torch_helpers::loadLSTM<float> (state_dict, "rec.", model.get<0>());
        RTNeural::torch_helpers::loadDense<float> (state_dict, "lin.", model.get<1>());
    }

    void process (std::span<float> data, float param)
    {
        alignas (16) float input[4] { 0.0f, param };
        for (auto& x : data)
        {
            input[0] = x;
            x += model.forward (input);
        }
    }
};

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
