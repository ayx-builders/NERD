# Copyright (C) 2020 Alteryx, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example pass through tool."""
import numpy
from ayx_plugin_sdk.core import (
    InputConnectionBase,
    Plugin,
    ProviderBase,
    register_plugin,
    FieldType
)
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter


class NERD(Plugin):
    def __init__(self, provider: ProviderBase):
        """Construct the AyxRecordProcessor."""
        self.config = provider.tool_config
        self.name = "NERD"
        self.provider = provider
        self.output_metadata = None
        self.output_anchor = self.provider.get_output_anchor("Output")
        self.splitter = SegtokSentenceSplitter()
        self.tagger_pos = SequenceTagger.load('pos')
        self.tagger_ner = SequenceTagger.load('ner-fast')

    def on_input_connection_opened(self, input_connection: InputConnectionBase) -> None:
        """Initialize the Input Connections of this plugin."""
        if input_connection.metadata is None:
            raise RuntimeError("Metadata must be set before setting containers.")

        output_metadata = input_connection.metadata.clone()
        output_metadata.fields = [i for i in output_metadata.fields if i.name != self.config['TextField']]
        output_metadata.add_field("Named Entity", FieldType.v_wstring, size=1073741823)
        output_metadata.add_field("Sentence", FieldType.int64)
        output_metadata.add_field("Sentence Position", FieldType.int64)
        output_metadata.add_field("Named Entity Type", FieldType.v_string, size=2147483647)
        self.output_metadata = output_metadata
        self.output_anchor.open(output_metadata)

    def on_record_packet(self, input_connection: InputConnectionBase) -> None:
        """Handle the record packet received through the input connection."""
        packet = input_connection.read()
        df_packet = packet.to_dataframe()

        df_packet['__NER__'] = df_packet[self.config['TextField']].apply(self.generate_ner)
        del df_packet[self.config['TextField']]
        df_packet = df_packet.explode('__NER__')
        df_packet['Named Entity'] = df_packet['__NER__'].apply(get_text)
        df_packet['Sentence'] = df_packet['__NER__'].apply(get_sentence)
        df_packet['Sentence Position'] = df_packet['__NER__'].apply(get_position)
        df_packet['Named Entity Type'] = df_packet['__NER__'].apply(get_type)
        del df_packet['__NER__']

        # X instantiate splitter
        # X instantiate tagging model (NER)
        # X Create function for NER
        # X run Apply using NER function
        # X Remove text column
        # X Explode dataframe
        # X Extract columns

        self.output_anchor.write(packet.from_dataframe(self.output_metadata, df_packet))

    def on_complete(self) -> None:
        return

    def generate_ner(self, text: str):
        if text is None:
            return []
        ner_data = []
        sentences_pos = self.splitter.split(text)
        self.tagger_pos.predict(sentences_pos)
        sentences_ner = self.splitter.split(text)
        self.tagger_ner.predict(sentences_ner)

        sentence_count = 1
        for sentence in sentences_pos:
            for entity in sentence.get_spans('pos'):
                ner_data.append(NerData(entity.text, sentence_count, entity.start_pos, entity.labels[0].value))
            sentence_count += 1
        sentence_count = 1
        for sentence in sentences_ner:
            for entity in sentence.get_spans('ner'):
                ner_data.append(NerData(entity.text, sentence_count, entity.start_pos, entity.labels[0].value))
            sentence_count += 1
        return ner_data


class NerData:
    def __init__(self, text, sentence, position, ner_type):
        self.text = text
        self.sentence = sentence
        self.position = position
        self.type = ner_type


def get_text(ner_data: NerData):
    if ner_data is numpy.nan:
        return None
    return ner_data.text


def get_sentence(ner_data: NerData):
    if ner_data is numpy.nan:
        return None
    return ner_data.sentence


def get_position(ner_data: NerData):
    if ner_data is numpy.nan:
        return None
    return ner_data.position


def get_type(ner_data: NerData):
    if ner_data is numpy.nan:
        return None
    return ner_data.type


AyxPlugin = register_plugin(NERD)
