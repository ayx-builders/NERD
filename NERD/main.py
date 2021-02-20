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
from ayx_plugin_sdk.core import (
    InputConnectionBase,
    Plugin,
    ProviderBase,
    register_plugin,
    FieldType
)
import flair


class NERD(Plugin):
    def __init__(self, provider: ProviderBase):
        """Construct the AyxRecordProcessor."""
        self.config = provider.tool_config
        self.name = "NERD"
        self.provider = provider
        self.output_anchor = self.provider.get_output_anchor("Output")

    def on_input_connection_opened(self, input_connection: InputConnectionBase) -> None:
        """Initialize the Input Connections of this plugin."""
        if input_connection.metadata is None:
            raise RuntimeError("Metadata must be set before setting containers.")

        output_metadata = input_connection.metadata.clone()
        output_metadata.fields = [i for i in output_metadata.fields if i.name != self.config['TextField']]
        output_metadata.add_field("Named Entity", FieldType.v_wstring, size=1073741823)
        output_metadata.add_field("Start Position", FieldType.int64)
        output_metadata.add_field("End Position", FieldType.int64)
        output_metadata.add_field("Named Entity Type", FieldType.v_string, size=2147483647)
        self.provider.io.info(f"{output_metadata.fields}")
        self.output_anchor.open(output_metadata)

    def on_record_packet(self, input_connection: InputConnectionBase) -> None:
        """Handle the record packet received through the input connection."""
        packet = input_connection.read()
        df_packet = packet.to_dataframe()

        # For each record, extract the Texts value
        # Run NER algorithm and identify named entities
        # Delete the Texts field from dataframe
        # Add our new fields
        # populate new fields from NER results

        self.output_anchor.write(packet.from_dataframe(df_packet))

    def on_complete(self) -> None:
        return


AyxPlugin = register_plugin(NERD)
