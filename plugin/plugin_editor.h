#pragma once

#include <chowdsp_gui/chowdsp_gui.h>

#include "neural_pruning_plugin.h"

struct Plugin_Editor : juce::AudioProcessorEditor
{
    explicit Plugin_Editor (Neural_Pruning_Plugin&);
    ~Plugin_Editor() override;

    void paint (juce::Graphics& g) override;

    chowdsp::ComponentArena<> arena {};
};
