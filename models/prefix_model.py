import torch
from torch import nn
from llm import LLM
from .panns import Wavegram_Logmel_Cnn14_Baseline, Wavegram_Logmel_Cnn14_Adaptation
from .mapping_networks import LocalGlobalTemporalMappingNetworks

pann_list = ["Wavegram_Logmel_Cnn14_Baseline", "Wavegram_Logmel_Cnn14_Adaptation"]
map_net_list = ['LocalGlobalTemporalMappingNetworks']


class PrefixModelForCaptioning(nn.Module):
    def __init__(
        self,
        pann_model: str,
        pann_pretrained_path: str,
        llm_model_repo: str,
        mapping_network: str,
        temporal_prefix_len: int,
        global_prefix_len: int,
    ):
        super().__init__(self)
        assert pann_model in pann_list, "Not exists PANN network"
        assert mapping_network in map_net_list, "Not exists Mapping Network"

        self.audio_encoder = eval(pann_model)()
        self.text_decoder = LLM(llm_model_repo=llm_model_repo)
        self.mapping_network = eval(mapping_network)(
            temporal_prefix_len, global_prefix_len
        )

        self.audio_encoder.load_pretrained(pann_pretrained_path)

    def freeze_parameters(self):
        print("Freezed Audio Encoder & LLM")
        self.audio_encoder.freeze()
        self.text_decoder.freeze()

    def forward(self, waveforms, labels):
        """
        waveforms (Float Tensor): [bs, len]
        labels (Long Tensor): [bs, len]
        """
        temporal_embeds, global_embeds = self.audio_encoder(waveforms)
        inputs_embeds = self.mapping_network(temporal_embeds, global_embeds)
        loss = self.text_decoder(inputs_embeds, labels)
        return loss

    def captioning(self, waveforms):
        """
        waveforms (tensor): [bs, len]
        """
        temporal_embeds, global_embeds = self.audio_encoder(waveforms)
        inputs_embeds = self.mapping_network(temporal_embeds, global_embeds)
        texts = self.text_decoder.generate(inputs_embeds)
        return texts
