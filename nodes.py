import os
import re
import torch
import hashlib
import requests
import folder_paths
import soundfile as sf

from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from rembg import new_session, remove
from torchvision.transforms.v2 import ToTensor, ToPILImage, Resize, CenterCrop
from bs4 import BeautifulSoup
from itertools import zip_longest

from .core.image.colorfix import adain_color_fix, wavelet_color_fix


##################################
# Global Variables and Functions #
##################################

class AnyType(str):
    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

to_tensor = ToTensor()
to_image = ToPILImage()

 
###########################
# ACE Nodes of Primitives #
###########################

class ACE_Integer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "int": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def execute(self, int):
        return (int,)
    
class ACE_Float:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float": ("FLOAT", {"default": 0.0}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def execute(self, float):
        return (float,)

class ACE_Text:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": '', "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def execute(self, text):
        return (text,)
    
class ACE_Seed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def execute(self, seed):
        return (seed,)
    

#####################
# ACE Nodes of Text #
#####################
    
class ACE_TextConcatenate:
    @ classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text1": ("STRING", {"multiline": True, "forceInput": True}),                
                "text2": ("STRING", {"multiline": True, "forceInput": True}), 
                "separator": ("STRING", {"default": ", ", "multiline": False}),                
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def execute(self, text1, text2, separator):
        return (text1 + separator + text2,)
    
class ACE_TextInputSwitch2Way:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("INT", {"default": 1, "min": 1, "max": 2}),
            },
            "optional": {
                "text1": ("STRING", {"default": '', "multiline": True, "forceInput": True}),
                "text2": ("STRING", {"default": '', "multiline": True, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def execute(self, input, text1='', text2=''):
        if input <= 1:
            return (text1,)
        else:
            return (text2,)  

class ACE_TextInputSwitch4Way:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("INT", {"default": 1, "min": 1, "max": 4}),
            },
            "optional": {
                "text1": ("STRING", {"default": '', "multiline": True, "forceInput": True}),
                "text2": ("STRING", {"default": '', "multiline": True, "forceInput": True}),
                "text3": ("STRING", {"default": '', "multiline": True, "forceInput": True}),
                "text4": ("STRING", {"default": '', "multiline": True, "forceInput": True}),  
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def execute(self, input, text1='', text2='', text3='', text4=''):
        if input <= 1:
            return (text1,)
        elif input == 2:
            return (text2,)
        elif input == 3:
            return (text3,)
        else:
            return (text4,)    

class ACE_TextInputSwitch8Way:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("INT", {"default": 1, "min": 1, "max": 8}),
            },
            "optional": {
                "text1": ("STRING", {"default": '', "multiline": True, "forceInput": True}),
                "text2": ("STRING", {"default": '', "multiline": True, "forceInput": True}),
                "text3": ("STRING", {"default": '', "multiline": True, "forceInput": True}),
                "text4": ("STRING", {"default": '', "multiline": True, "forceInput": True}),  
                "text5": ("STRING", {"default": '', "multiline": True, "forceInput": True}),
                "text6": ("STRING", {"default": '', "multiline": True, "forceInput": True}),
                "text7": ("STRING", {"default": '', "multiline": True, "forceInput": True}),
                "text8": ("STRING", {"default": '', "multiline": True, "forceInput": True}),  
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def execute(self, input, text1='', text2='', text3='', text4='', text5='', text6='', text7='', text8=''):
        if input <= 1:
            return (text1,)
        elif input == 2:
            return (text2,)
        elif input == 3:
            return (text3,)
        elif input == 4:
            return (text4,)
        elif input == 5:
            return (text5,)
        elif input == 6:
            return (text6,)
        elif input == 7:
            return (text7,)
        else:
            return (text8,)  
        
class ACE_TextList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":{
                "list_text": ("STRING", {"default": '', "multiline": True}),              
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"
    
    def execute(self, list_text):
        lines = list_text.split('\n')
        list_out = [x.strip() for x in lines if x.strip()]
        return (list_out,)
    
class ACE_TextPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def execute(self, text):
        return {"ui": {"text": text}, "result": (text,)}
    
class ACE_TextSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":{
                "list_text": ("STRING", {"default": '', "multiline": True}),    
                "select": ("INT", {"default": 0, "min": 0}),      
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"
    
    def execute(self, list_text, select):
        lines = list_text.split('\n')
        list_out = [x.strip() for x in lines if x.strip()]
        select = max(min(select, len(list_out)-1), 0)
        return (list_out[select],)
    
class ACE_TextToResolution:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":{
                "text": ("STRING", {"default": '', "forceInput": True}),
            }
        }

    RETURN_TYPES = ("INT","INT",)
    RETURN_NAMES = ("WIDTH","HEIGHT",)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"
    
    def execute(self, text):
        width, height = text.strip().split(" ")[0].split("x")
        width, height = int(width), int(height)
        return (width,height,)
    
class ACE_TextTranslate:
    def __init__(self):
        self.model_checkpoint = None
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        supported_lang = [
            'en',  # 英语
            'zh',  # 中文
            'es',  # 西班牙语
            'hi',  # 印度语
            'ar',  # 阿拉伯语
            'pt',  # 葡萄牙语
            'ru',  # 俄语
            'ja',  # 日语
            'de',  # 德语
            'fr',  # 法语
            'ko',  # 韩语
            'it',  # 意大利语
            'nl',  # 荷兰语
            'tr',  # 土耳其语
            'sv',  # 瑞典语
            'pl',  # 波兰语
            'th',  # 泰语
            'vi',  # 越南语
            'id',  # 印尼语
            'el',  # 希腊语
            'cs',  # 捷克语
            'da',  # 丹麦语
            'fi',  # 芬兰语
            'hu',  # 匈牙利语
            'no',  # 挪威语
            'ro',  # 罗马尼亚语
            'sk',  # 斯洛伐克语
            'uk',  # 乌克兰语
        ]
        return {
            "required":{
                "text": ("STRING", {"default": '', "multiline": True}),
                "from_lang": (supported_lang, {"default": 'en'}),
                "to_lang": (supported_lang, {"default": 'en'}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"
    
    def execute(self, text, from_lang, to_lang):
        if from_lang == to_lang:
            return (text,)
        
        model_id = f'Helsinki-NLP/opus-mt-{from_lang}-{to_lang}'
        model_checkpoint = os.path.join(folder_paths.models_dir, 'prompt_generator', os.path.basename(model_id))

        if not os.path.exists(model_checkpoint):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_id, local_dir=model_checkpoint, local_dir_use_symlinks=False)

        if self.model_checkpoint != model_checkpoint:
            self.model_checkpoint = model_checkpoint
            self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(self.device).eval()

        with torch.no_grad():
            texts = [x.strip() for x in text.split("\n") if x.strip()]
            encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            sequences = self.model.generate(**encoded)
            translation = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
            translation_text = "\n".join([x.rstrip('.') for x in translation])
            return (translation_text,)
        
class ACE_TextGoogleTranslate:
    @classmethod
    def INPUT_TYPES(cls):
        supported_lang = [
            'en',  # 英语
            'zh',  # 中文
            'es',  # 西班牙语
            'hi',  # 印度语
            'ar',  # 阿拉伯语
            'pt',  # 葡萄牙语
            'ru',  # 俄语
            'ja',  # 日语
            'de',  # 德语
            'fr',  # 法语
            'ko',  # 韩语
            'it',  # 意大利语
            'nl',  # 荷兰语
            'tr',  # 土耳其语
            'sv',  # 瑞典语
            'pl',  # 波兰语
            'th',  # 泰语
            'vi',  # 越南语
            'id',  # 印尼语
            'el',  # 希腊语
            'cs',  # 捷克语
            'da',  # 丹麦语
            'fi',  # 芬兰语
            'hu',  # 匈牙利语
            'no',  # 挪威语
            'ro',  # 罗马尼亚语
            'sk',  # 斯洛伐克语
            'uk',  # 乌克兰语
        ]
        return {
            "required":{
                "text": ("STRING", {"default": '', "multiline": True}),
                "from_lang": (supported_lang, {"default": 'en'}),
                "to_lang": (supported_lang, {"default": 'en'}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"
    
    def execute(self, text, from_lang, to_lang):
        if from_lang == to_lang:
            return (text,)
        
        response = requests.get(f'https://translate.google.com/m?sl={from_lang}&tl={to_lang}&q={text}')

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            element = soup.find('div', {"class": "result-container"})

            if element:
                translation_text = element.get_text(strip=True)
                return (translation_text,)

        return (text,)
    

######################
# ACE Nodes of Image #
######################

class ACE_ImageConstrain:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "max_width": ("INT", {"default": 1024, "min": 0}),
                "max_height": ("INT", {"default": 1024, "min": 0}),
                "min_width": ("INT", {"default": 0, "min": 0}),
                "min_height": ("INT", {"default": 0, "min": 0}),
                "crop_if_required": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def execute(self, images, max_width, max_height, min_width, min_height, crop_if_required):
        images = images.permute([0,3,1,2])
        output = []

        for image in images:
            image = to_image(image)

            current_width, current_height = image.size
            aspect_ratio = current_width / current_height

            target_width = min(max(current_width, min_width), max_width)
            target_height = min(max(current_height, min_height), max_height)

            if crop_if_required:
                if target_width / target_height < aspect_ratio:
                    resize_width, resize_height = int(target_height * aspect_ratio), int(target_height)
                else:
                    resize_width, resize_height = int(target_width), int(target_width / aspect_ratio)
                image = Resize((resize_height, resize_width))(image)
                image = CenterCrop((target_height, target_width))(image)
            else:
                if target_width / target_height > aspect_ratio:
                    target_width, target_height = int(target_height * aspect_ratio), int(target_height)
                else:
                    target_width, target_height = int(target_width), int(target_width / aspect_ratio)
                image = Resize((max(target_height, min_height), max(target_width, min_width)))(image)

            output.append(to_tensor(image))

        output = torch.stack(output, dim=0)
        output = output.permute([0,2,3,1])
                
        return (output[:, :, :, :3],)
    
class ACE_ImageRemoveBackground:
    def __init__(self):
        U2NET_HOME=os.path.join(folder_paths.models_dir, "rembg")
        os.environ["U2NET_HOME"] = U2NET_HOME

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model": (["u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta", "isnet-general-use", "isnet-anime", "sam"],),
            },
        }

    RETURN_TYPES = ("IMAGE","MASK",)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def execute(self, images, model):
        rembg_session = new_session(model, providers=['CPUExecutionProvider'])

        images = images.permute([0,3,1,2])
        output = []
        
        for image in images:
            image = to_image(image)
            image = remove(image, session=rembg_session)
            output.append(to_tensor(image))

        output = torch.stack(output, dim=0)
        output = output.permute([0,2,3,1])
        mask = output[:, :, :, 3] if output.shape[3] == 4 else torch.ones_like(output[:, :, :, 0])

        return(output[:, :, :, :3], mask,)
    
class ACE_ImageColorFix:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "color_map_images": ("IMAGE",),
                "color_fix": (
                    [
                        "Wavelet",
                        "AdaIN",
                    ],
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def execute(self, images, color_map_images, color_fix):
        color_fix_func = (
            wavelet_color_fix if color_fix == "Wavelet" else adain_color_fix
        )

        images = images.permute([0,3,1,2])
        color_map_images = color_map_images.permute([0,3,1,2])
        output = []

        last_element = images[-1] if len(images) < len(color_map_images) else color_map_images[-1]
        
        for image, color_map_image in zip_longest(images, color_map_images, fillvalue=last_element):
            result_image = color_fix_func(to_image(image), to_image(color_map_image))
            output.append(to_tensor(result_image))

        output = torch.stack(output, dim=0)
        output = output.permute([0,2,3,1])
                
        return (output[:, :, :, :3],)

class ACE_ImageQA:
    def __init__(self):
        self.model_checkpoint = None
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.bf16_support = torch.cuda.is_available() and torch.cuda.get_device_capability(self.device)[0] >= 8

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"default": '', "multiline": True}),
                "model": (["moondream2", "MiniCPM-V-2"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def execute(self, image, text, model):
        if model == "moondream2":
            model_id = "vikhyatk/moondream2"
        elif model == "MiniCPM-V-2":
            model_id = "openbmb/MiniCPM-V-2"
        else:
            raise Exception(f'Model "{model}" is not supported')
        
        model_checkpoint = os.path.join(folder_paths.models_dir, 'prompt_generator', os.path.basename(model_id))

        if not os.path.exists(model_checkpoint):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_id, local_dir=model_checkpoint, local_dir_use_symlinks=False)

        if self.model_checkpoint != model_checkpoint:
            self.model_checkpoint = model_checkpoint
            if model == "moondream2":
                self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(model_checkpoint, trust_remote_code=True)
                self.model = self.model.to(self.device).eval()
            elif model == "MiniCPM-V-2":
                self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(model_checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16)
                self.model = self.model.to(self.device, dtype=torch.bfloat16 if self.bf16_support else torch.float16).eval()

        with torch.no_grad():
            image = to_image(image.permute([0,3,1,2])[0]).convert("RGB")

            if model == "moondream2":
                encoded_image = self.model.encode_image(image)
                result = self.model.answer_question(encoded_image, text, self.tokenizer)
            elif model == "MiniCPM-V-2":
                result, context, _ = self.model.chat(
                    image=image,
                    msgs=[{'role': 'user', 'content': text}],
                    context=None,
                    tokenizer=self.tokenizer,
                    sampling=True
                )
            return (result,)
    

######################
# ACE Nodes of Audio #
######################
    
class ACE_AudioLoad:
    @classmethod
    def INPUT_TYPES(s):
        audio_extensions = ["wav", "mp3", "flac"]
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.lower().split('.')
                if len(file_parts) > 1 and (file_parts[-1] in audio_extensions):
                    files.append(f)
        return {
            "required": {
                "audio": (sorted(files),),
            },
        }

    RETURN_TYPES = (any, "INT",)
    RETURN_NAMES = ("AUDIO", "SAMPLE_RATE",)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def execute(self, audio):
        file = folder_paths.get_annotated_filepath(audio)
        ext = file.lower().split('.')[-1] if '.' in file else 'null'

        if ext in ["wav", "mp3", "flac"]:
            audio_samples, sample_rate =sf.read(file)
        else:
            raise Exception(f'File format "{ext}" is not supported')

        return (list(audio_samples), sample_rate)
    
    @classmethod
    def IS_CHANGED(self, audio, **kwargs):
        audio_path = folder_paths.get_annotated_filepath(audio)
        m = hashlib.sha256()
        with open(audio_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(self, audio, **kwargs):
        if not folder_paths.exists_annotated_filepath(audio):
            return "Invalid audio file: {}".format(audio)
        return True
    
class ACE_AudioSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (any, {"forceInput": True}),
                "sample_rate": ("INT", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "extension": (["wav", "mp3", "flac"], {"default": "wav"}),
            }
        }
    
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def execute(self, audio, sample_rate, filename_prefix, extension):
        subfolder = os.path.dirname(os.path.normpath(filename_prefix))
        filename = os.path.basename(os.path.normpath(filename_prefix))

        output_dir = folder_paths.get_output_directory()
        full_output_folder = os.path.join(output_dir, subfolder)

        max_counter = 0
        matcher = re.compile(f"{re.escape(filename)}_(\d+)\D*\..+", re.IGNORECASE)
        for existing_file in os.listdir(full_output_folder):
            match = matcher.fullmatch(existing_file)
            if match:
                max_counter = max(max_counter, int(match.group(1)))
        counter = max_counter + 1

        audio_path = os.path.join(full_output_folder, f"{filename}_{counter:05}.{extension}")
        sf.write(audio_path, audio, sample_rate)

        return ()
    
class ACE_AudioPlay:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["always", "on empty queue"], {}),
                "volume": ("FLOAT", {"min": 0, "max": 1, "step": 0.1, "default": 0.5}),
                "audio": (any, {"forceInput": True}),
                "sample_rate": ("INT", {"forceInput": True}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = (any,)
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def IS_CHANGED(self, **kwargs):
        return float("NaN")

    def execute(self, mode, volume, audio, sample_rate):
        return {"ui": {"audio": audio, "sample_rate": sample_rate}, "result": (any,)}
    

#########################
# ACE Nodes for ComfyUI #
#########################

NODE_CLASS_MAPPINGS = {
    "ACE_Integer"               : ACE_Integer,
    "ACE_Float"                 : ACE_Float,
    "ACE_Text"                  : ACE_Text,
    "ACE_Seed"                  : ACE_Seed,

    "ACE_TextConcatenate"       : ACE_TextConcatenate,
    "ACE_TextInputSwitch2Way"   : ACE_TextInputSwitch2Way,
    "ACE_TextInputSwitch4Way"   : ACE_TextInputSwitch4Way,
    "ACE_TextInputSwitch8Way"   : ACE_TextInputSwitch8Way,
    "ACE_TextList"              : ACE_TextList,
    "ACE_TextPreview"           : ACE_TextPreview,
    "ACE_TextSelector"          : ACE_TextSelector,
    "ACE_TextToResolution"      : ACE_TextToResolution,
    "ACE_TextTranslate"         : ACE_TextTranslate,
    "ACE_TextGoogleTranslate"   : ACE_TextGoogleTranslate,

    "ACE_ImageConstrain"        : ACE_ImageConstrain,
    "ACE_ImageRemoveBackground" : ACE_ImageRemoveBackground,
    "ACE_ImageColorFix"         : ACE_ImageColorFix,
    "ACE_ImageQA"               : ACE_ImageQA,

    "ACE_AudioLoad"             : ACE_AudioLoad,
    "ACE_AudioSave"             : ACE_AudioSave,
    "ACE_AudioPlay"             : ACE_AudioPlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ACE_Integer"               : "🅐 Integer",
    "ACE_Float"                 : "🅐 Float",
    "ACE_Text"                  : "🅐 Text",
    "ACE_Seed"                  : "🅐 Seed",

    "ACE_TextConcatenate"       : "🅐 Text Concatenate",
    "ACE_TextInputSwitch2Way"   : "🅐 Text Input Switch (2 way)",
    "ACE_TextInputSwitch4Way"   : "🅐 Text Input Switch (4 way)",
    "ACE_TextInputSwitch8Way"   : "🅐 Text Input Switch (8 way)",
    "ACE_TextList"              : "🅐 Text List",
    "ACE_TextPreview"           : "🅐 Text Preview",
    "ACE_TextSelector"          : "🅐 Text Selector",
    "ACE_TextToResolution"      : "🅐 Text To Resolution",
    "ACE_TextTranslate"         : "🅐 Text Translate",
    "ACE_TextGoogleTranslate"   : "🅐 Text Google Translate",

    "ACE_ImageConstrain"        : "🅐 Image Constrain",
    "ACE_ImageRemoveBackground" : "🅐 Image Remove Background",
    "ACE_ImageColorFix"         : "🅐 Image Color Fix",
    "ACE_ImageQA"               : "🅐 Image Question Answering",

    "ACE_AudioLoad"             : "🅐 Audio Load",
    "ACE_AudioSave"             : "🅐 Audio Save",
    "ACE_AudioPlay"             : "🅐 Audio Play",
}