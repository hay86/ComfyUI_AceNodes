import os
import re
import torch
import hashlib
import folder_paths
import numpy as np

from PIL import Image
from datetime import datetime
from torchvision.transforms.v2 import ToTensor, ToPILImage


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
                "int": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 1}),
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
                "float": ("FLOAT", {"default": 0.0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.001}),
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
                "text": ("STRING", {"default": '512x512', "forceInput": True}),
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
        self.device = (
            torch.device("mps") 
            if torch.backends.mps.is_available()
            else torch.device("cuda") 
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

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
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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
        
        import requests
        response = requests.get(f'https://translate.google.com/m?sl={from_lang}&tl={to_lang}&q={text}')

        if response.status_code == 200:
            from bs4 import BeautifulSoup
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
            "optional": {
                "crop_position": (["center", "top", "bottom", "left", "right"], {"default": "center"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def execute(self, images, max_width, max_height, min_width, min_height, crop_if_required, crop_position="center"):
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
                image = image.resize((resize_width, resize_height), resample=Image.Resampling.LANCZOS)
                x0, y0 = max((resize_width-target_width)/2, 0), max((resize_height-target_height)/2, 0)
                if crop_position == "top":
                    y0 = 0
                elif crop_position == "bottom":
                    y0 = resize_height - target_height
                elif crop_position == "left":
                    x0 = 0
                elif crop_position == "right":
                    x0 = resize_width - target_width
                image = image.crop((x0, y0, x0+target_width, y0+target_height))
            else:
                if target_width / target_height > aspect_ratio:
                    target_width, target_height = int(target_height * aspect_ratio), int(target_height)
                else:
                    target_width, target_height = int(target_width), int(target_width / aspect_ratio)
                resize_width, resize_height = max(target_width, min_width), max(target_height, min_height)
                image = image.resize((resize_width, resize_height), resample=Image.Resampling.LANCZOS)

            output.append(to_tensor(image))

        output = torch.stack(output, dim=0)
        output = output.permute([0,2,3,1])
                
        return (output[:, :, :, :3],)
    
class ACE_ImageRemoveBackground:
    def __init__(self):
        self.model_dir = os.path.join(folder_paths.models_dir, "rembg")
        os.environ["U2NET_HOME"] = self.model_dir 
        self.model_checkpoint = None
        self.model = None
        self.device = (
            torch.device("mps") 
            if torch.backends.mps.is_available()
            else torch.device("cuda") 
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model": (["briarmbg", "u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta", "isnet-general-use", "isnet-anime", "sam"],),
            },
        }

    RETURN_TYPES = ("IMAGE","MASK",)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def execute(self, images, model):
        if model == "briarmbg":
            model_checkpoint = os.path.join(self.model_dir, 'briarmbg.pth')

            from .core.image.briarmbg import BriaRMBG
            if self.model_checkpoint != model_checkpoint:
                if not os.path.exists(model_checkpoint):
                    from huggingface_hub import hf_hub_download
                    hf_hub_download(repo_id='briaai/RMBG-1.4', filename='model.pth', local_dir=self.model_dir)
                    os.rename(os.path.join(self.model_dir, 'model.pth'), model_checkpoint)
                
                self.model_checkpoint = model_checkpoint
                self.model = BriaRMBG()
                self.model.load_state_dict(torch.load(self.model_checkpoint, map_location=self.device))
                self.model.to(self.device)
                self.model.eval() 

            images = images.permute([0,3,1,2])
            processed_images = []
            processed_masks = []

            import torch.nn.functional as F
            from torchvision.transforms.v2.functional import normalize
            for image in images:
                orig_image = to_image(image)
                w,h = orig_image.size
                image = orig_image.convert('RGB')
                image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
                im_np = np.array(image)
                im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2,0,1)
                im_tensor = torch.unsqueeze(im_tensor,0)
                im_tensor = torch.divide(im_tensor,255.0)
                im_tensor = normalize(im_tensor,[0.5,0.5,0.5],[1.0,1.0,1.0])
                im_tensor = im_tensor.to(self.device)

                result = self.model(im_tensor)
                result = torch.squeeze(F.interpolate(result[0][0], size=(h,w), mode='bilinear') ,0)
                ma = torch.max(result)
                mi = torch.min(result)
                result = (result-mi)/(ma-mi)    
                im_array = (result*255).cpu().data.numpy().astype(np.uint8)
                pil_im = Image.fromarray(np.squeeze(im_array))
                new_im = Image.new("RGB", pil_im.size, (0,0,0))
                new_im.paste(orig_image, mask=pil_im)

                processed_images.append(to_tensor(new_im))
                processed_masks.append(to_tensor(pil_im))

            new_ims = torch.stack(processed_images, dim=0)
            new_masks = torch.stack(processed_masks, dim=0)
            new_ims = new_ims.permute([0,2,3,1])

            return (new_ims, new_masks,)

        else:
            from rembg import new_session, remove
            if self.model_checkpoint != model:
                self.model_checkpoint = model
                self.model = new_session(self.model_checkpoint, providers=['CPUExecutionProvider'])

            images = images.permute([0,3,1,2])
            output = []
            
            for image in images:
                image = to_image(image)
                image = remove(image, session=self.model)
                output.append(to_tensor(image))

            output = torch.stack(output, dim=0)
            output = output.permute([0,2,3,1])
            mask = output[:, :, :, 3] if output.shape[3] == 4 else torch.ones_like(output[:, :, :, 0])

            return (output[:, :, :, :3], mask,)
    
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
        from .core.image.colorfix import adain_color_fix, wavelet_color_fix
        color_fix_func = (
            wavelet_color_fix if color_fix == "Wavelet" else adain_color_fix
        )

        images = images.permute([0,3,1,2])
        color_map_images = color_map_images.permute([0,3,1,2])
        output = []

        last_element = images[-1] if len(images) < len(color_map_images) else color_map_images[-1]
        
        from itertools import zip_longest
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
        self.device = (
            torch.device("mps") 
            if torch.backends.mps.is_available()
            else torch.device("cuda") 
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.bf16_support = (
            torch.backends.mps.is_available() or 
            (torch.cuda.is_available() and torch.cuda.get_device_capability(self.device)[0] >= 8)
        )

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
                from transformers import AutoTokenizer, AutoModelForCausalLM
                self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(model_checkpoint, trust_remote_code=True)
                self.model = self.model.to(self.device).eval()
            elif model == "MiniCPM-V-2":
                from transformers import AutoTokenizer, AutoModel
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
        
class ACE_ImageLoadFromCloud:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filepath": ("STRING", {"default": ''}),
                "bucket": ("STRING", {"default": ''}),
                "region": ("STRING", {"default": ''}),
                "cloud": (["aws-s3","aliyun-oss"],),
                "access_key_id": ("STRING", {"default": ''}),
                "access_key_secret": ("STRING", {"default": ''}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def execute(self, filepath, bucket, region, cloud, access_key_id, access_key_secret):
        save_path = os.path.join(folder_paths.temp_directory, bucket, filepath)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if cloud == "aws-s3":
            import boto3
            s3 = boto3.client('s3', 
                              aws_access_key_id=access_key_id, 
                              aws_secret_access_key=access_key_secret, 
                              region_name=region)
            s3.download_file(bucket, filepath, save_path)
        elif cloud == "aliyun-oss":
            import oss2
            oss_auth = oss2.Auth(access_key_id, access_key_secret)
            oss_bucket = oss2.Bucket(oss_auth, region, bucket)
            oss_bucket.get_object_to_file(filepath, save_path)
        else:
            raise Exception(f'Cloud "{cloud}" is not supported')
            
        image = f"{bucket}/{filepath} [temp]"
        
        image_path = folder_paths.get_annotated_filepath(image)
        from PIL import ImageSequence, ImageOps
        img = Image.open(image_path)
        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

class ACE_ImageSaveToCloud:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filepath": ("STRING", {"default": 'ComfyUI_{0:05d}_{1:%Y-%m-%d_%H:%M:%S}.png'}),
                "bucket": ("STRING", {"default": ''}),
                "region": ("STRING", {"default": ''}),
                "cloud": (["aws-s3","aliyun-oss"],),
                "access_key_id": ("STRING", {"default": ''}),
                "access_key_secret": ("STRING", {"default": ''}),
            },
        }
    
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def execute(self, images, filepath, bucket, region, cloud, access_key_id, access_key_secret):
        now = datetime.now()
        results = []
        files_to_upload = []
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            filepath_formated = filepath.format(batch_number, now)
            save_path = os.path.join(folder_paths.temp_directory, bucket, filepath_formated)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            img.save(save_path, compress_level=1)

            file, subfolder = os.path.basename(filepath_formated), os.path.join(bucket, os.path.dirname(filepath_formated))
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": "temp"
            })
            files_to_upload.append((save_path, filepath_formated))

        if cloud == "aws-s3":
            import boto3
            s3 = boto3.client('s3', 
                              aws_access_key_id=access_key_id, 
                              aws_secret_access_key=access_key_secret, 
                              region_name=region)
            for filename, objectname in files_to_upload:
                try:
                    s3.upload_file(filename, bucket, objectname)
                except Exception as e:
                    print(f'An error occurred: {e}')
        elif cloud == "aliyun-oss":
            import oss2
            oss_auth = oss2.Auth(access_key_id, access_key_secret)
            oss_bucket = oss2.Bucket(oss_auth, region, bucket)
            for filename, objectname in files_to_upload:
                try:
                    oss_bucket.put_object_from_file(objectname, filename)
                except Exception as e:
                    print(f'An error occurred: {e}')
        else:
            raise Exception(f'Cloud "{cloud}" is not supported')

        return { "ui": { "images": results } }
    
class ACE_ImageGetSize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("WIDTH","HEIGHT",)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def execute(self, image):
        return (image.shape[2], image.shape[1],)
    
class ACE_ImageFaceCrop:
    def __init__(self):
        self.model_name = None
        self.model = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (["retinaface", "insightface"],),
                "crop_width": ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
                "crop_height": ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE","MASK","BOOLEAN",)
    RETURN_NAMES = ("IMAGE","MASK","FACE_DETECTED",)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def execute(self, image, model, crop_width, crop_height):
        image = to_image(image.permute([0,3,1,2])[0]).convert("RGB")
        im_np = np.array(image)

        face_bboxes = []
        face_images = []
        face_masks = []

        if model == 'retinaface':
            from retinaface import RetinaFace
            if self.model_name != model:
                self.model_name = model
                self.model = RetinaFace.build_model()

            faces = RetinaFace.detect_faces(im_np, model=self.model)
            if faces:
                for face in faces.values():
                    face_bboxes.append(face['facial_area'])
        elif model == 'insightface':
            from insightface.app import FaceAnalysis
            if self.model_name != model:
                self.model_name = model
                self.model = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], root=os.path.join(folder_paths.models_dir, 'insightface'))
                self.model.prepare(ctx_id=0)

            faces = self.model.get(im_np)
            if faces:
                for face in faces:
                    face_bboxes.append(face.bbox.astype(int))
        else:
            raise Exception(f'Model "{model}" is not supported')
        
        if face_bboxes:
            face_bboxes = sorted(face_bboxes, key=lambda x: -(x[2]-x[0])*(x[3]-x[1])) # sort by area
            for bbox in face_bboxes:
                x1, y1, x2, y2 = bbox
                width, height = x2-x1, y2-y1
                if width / height > crop_width / crop_height:
                    new_height = int(width * crop_height / crop_width)
                    y_offset = int((new_height - height) / 2)
                    y1, y2 = y1-y_offset, y2+y_offset
                else:
                    new_width = int(height * crop_width / crop_height)
                    x_offset = int((new_width - width) / 2)
                    x1, x2 = x1-x_offset, x2+x_offset
                
                face_image = image.crop((x1, y1, x2, y2)).resize((crop_width, crop_height), Image.Resampling.LANCZOS)
                face_images.append(to_tensor(face_image))

                face_mask = torch.zeros(image.size)
                face_mask[max(x1,0):min(x2,image.size[0]-1), max(y1,0):min(y2,image.size[0]-1)] = 1
                face_masks.append(face_mask)
        else:
            face_image = torch.zeros((3, crop_width, crop_height))
            face_images.append(to_tensor(face_image))
        
            face_mask = torch.zeros(image.size)
            face_masks.append(face_mask)

        output_images = torch.stack(face_images, dim=0)
        output_images = output_images.permute([0,2,3,1])
        output_masks = torch.stack(face_masks, dim=0)
        output_masks = output_masks.permute([0,2,1])
        
        return (output_images, output_masks, len(face_bboxes)>0)


#####################
# ACE Nodes of Mask #
#####################
    
class ACE_MaskBlur:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "radius": ("INT", { "default": 5, "min": 0, "max": 256, "step": 1, }),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def execute(self, mask, radius):
        if radius == 0:
            return (mask,)

        if radius % 2 == 0:
            radius += 1

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        from torchvision.transforms.v2.functional import gaussian_blur
        mask = gaussian_blur(mask.unsqueeze(1), kernel_size=int(6 * radius + 1), sigma=radius).squeeze(1)

        return(mask,)


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
            import soundfile as sf
            audio_samples, sample_rate =sf.read(file)
        else:
            raise Exception(f'File format "{ext}" is not supported')

        return (audio_samples.tolist(), sample_rate)
    
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
        import soundfile as sf
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
    

#######################
# ACE Nodes of Others #
#######################
    
class ACE_ExpressionEval:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "a": (any, {"default": ""}),
                "b": (any, {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING","INT","FLOAT",)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def execute(self, value, a='', b=''):
        result = eval(value, {'a':a, 'b':b})
        try:
            result_int = round(int(result))
        except:
            result_int = 0
        try:
            result_float = round(float(result), 4)
        except:
            result_float = 0
        return (str(result), result_int, result_float)
    
class ACE_AnyInputSwitchBool:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bool": (any,),
                "any_if_true": (any,),
                "any_if_false": (any,),
            },
        }

    RETURN_TYPES = (any,)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def execute(self, bool, any_if_true, any_if_false):
        return (any_if_true if bool else any_if_false,)
    
class ACE_AnyInputToAny:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any,),
            },
        }

    RETURN_TYPES = (any,)
    FUNCTION = "execute"
    CATEGORY = "Ace Nodes"

    def execute(self, any):
        return (any,)


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
    "ACE_ImageLoadFromCloud"    : ACE_ImageLoadFromCloud,
    "ACE_ImageSaveToCloud"      : ACE_ImageSaveToCloud,
    "ACE_ImageGetSize"          : ACE_ImageGetSize,
    "ACE_ImageFaceCrop"         : ACE_ImageFaceCrop,

    "ACE_MaskBlur"              : ACE_MaskBlur,

    "ACE_AudioLoad"             : ACE_AudioLoad,
    "ACE_AudioSave"             : ACE_AudioSave,
    "ACE_AudioPlay"             : ACE_AudioPlay,

    "ACE_Expression_Eval"       : ACE_ExpressionEval,
    "ACE_AnyInputSwitchBool"    : ACE_AnyInputSwitchBool,
    "ACE_AnyInputToAny"         : ACE_AnyInputToAny,
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
    "ACE_ImageLoadFromCloud"    : "🅐 Image Load From Cloud",
    "ACE_ImageSaveToCloud"      : "🅐 Image Save To Cloud",
    "ACE_ImageGetSize"          : "🅐 Image Get Size",
    "ACE_ImageFaceCrop"         : "🅐 Image Face Crop",

    "ACE_MaskBlur"              : "🅐 Mask Blur",

    "ACE_AudioLoad"             : "🅐 Audio Load",
    "ACE_AudioSave"             : "🅐 Audio Save",
    "ACE_AudioPlay"             : "🅐 Audio Play",

    "ACE_Expression_Eval"       : "🅐 Expression Eval",
    "ACE_AnyInputSwitchBool"    : "🅐 Any Input Switch (bool)",
    "ACE_AnyInputToAny"         : "🅐 Any Input To Any",
}