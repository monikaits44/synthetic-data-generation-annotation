"""
Prompt Enhancement Module
=========================
LLM-driven prompt enhancement using Ollama for Flux and Stable Diffusion models.
Implements token-aware enhancement with allowed object validation.
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple
import ollama
import tiktoken

# ==================== CONFIGURATION ====================
class EnhancementConfig:
    """Configuration for prompt enhancement"""
    MODEL_NAME = "llama2:latest"
    MAX_RETRIES = 3
    
    # Token limits per model
    FLUX_TOKEN_LIMIT = 512
    STABLE_DIFFUSION_TOKEN_LIMIT = 72
    
    # Allowed object classes (extendable)
    ALLOWED_OBJECTS = {
        "traffic_cone", "fire_extinguisher", "cardbox", "pallet",
        "bottle", "vase", "chair", "monitor", "plastic_box",
        "shoe", "apple", "robotic_arm", "cup", "screw_driver",
        "plier", "hammer"
    }

# ==================== TOKEN HANDLING ====================
class TokenManager:
    """Manages token counting and truncation"""
    
    def __init__(self):
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if not self.tokenizer:
            # Fallback: approximate (1 token ≈ 4 characters)
            return len(text) // 4
        return len(self.tokenizer.encode(text))
    
    def encode_tokens(self, text: str) -> List[int]:
        """Encode text to tokens"""
        if not self.tokenizer:
            return []
        return self.tokenizer.encode(text)
    
    def decode_tokens(self, tokens: List[int]) -> str:
        """Decode tokens to text"""
        if not self.tokenizer:
            return ""
        return self.tokenizer.decode(tokens)
    
    def truncate_to_limit(self, text: str, token_limit: int) -> str:
        """Truncate text to fit within token limit"""
        if not self.tokenizer:
            char_limit = token_limit * 4
            return text[:char_limit] if len(text) > char_limit else text
        
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= token_limit:
            return text
        
        truncated_tokens = tokens[:token_limit]
        truncated_text = self.tokenizer.decode(truncated_tokens)
        
        # Clean up incomplete words
        words = truncated_text.split()
        if len(words) > 1:
            last_word = words[-1]
            if not last_word.endswith(('.', ',', '!', '?', ':')):
                words = words[:-1]
        
        return ' '.join(words)

# ==================== OBJECT NAME NORMALIZATION ====================
class ObjectNormalizer:
    """Normalizes and validates object names"""
    
    @staticmethod
    def get_object_mappings() -> Dict[str, str]:
        """Comprehensive object name mappings"""
        return {
            # Screwdriver
            "screwdriver": "screw_driver",
            "screwdrivers": "screw_driver",
            "screw driver": "screw_driver",
            
            # Tools
            "tool": "hammer",
            "tools": "hammer",
            "hand_tool": "hammer",
            
            # Boxes
            "cardboard": "cardbox",
            "box": "cardbox",
            "cardboard_box": "cardbox",
            "plastic_box": "plastic_box",
            
            # Others
            "robotic_arm": "robotic_arm",
            "robot_arm": "robotic_arm",
            "fire_extinguisher": "fire_extinguisher",
            "traffic_cone": "traffic_cone",
            "cone": "traffic_cone",
            "pallet": "pallet",
            "wooden_pallet": "pallet",
            "pliers": "plier",
            "plier": "plier",
            "hammer": "hammer",
            "bottle": "bottle",
            "shoe": "shoe",
            "chair": "chair",
            "cup": "cup",
            "apple": "apple",
            "vase": "vase",
            "monitor": "monitor",
            "screen": "monitor",
        }
    
    @staticmethod
    def normalize(obj_name: str) -> str:
        """Normalize object name using mapping rules"""
        mappings = ObjectNormalizer.get_object_mappings()
        obj_clean = obj_name.strip().lower().replace(" ", "_").replace("-", "_")
        return mappings.get(obj_clean, obj_clean)

# ==================== SYSTEM PROMPTS ====================
class SystemPrompts:
    """System prompts for different models"""
    
    @staticmethod
    def get_flux_prompt(allowed_objects: str) -> str:
        """System prompt for Flux enhancement"""
        return f"""You are a prompt enhancement specialist. Your task is to analyze a user prompt and create a JSON response.

CRITICAL RULES:
1. Return ONLY valid JSON, no other text
2. Use exact keys: Summary, Attributes
3. Attributes must have exactly these keys: Object Class, Environment, Lighting, Texture, Material, Distinct Features, Technical Details
4. Object Class must use objects from: {allowed_objects}
5. All values must be strings
6. No escape sequences inside values
7. Keep it concise and factual

Example format:
{{"Summary": "A robotic arm in an industrial warehouse", "Attributes": {{"Object Class": "robotic_arm", "Environment": "Industrial warehouse", "Lighting": "Bright overhead lights", "Texture": "Metallic", "Material": "Steel and aluminum", "Distinct Features": "Articulated joints", "Technical Details": "Wide angle shot"}}}}

Remember: ONLY return JSON, nothing else."""
    
    @staticmethod
    def get_stable_diffusion_prompt(allowed_objects: str) -> str:
        """System prompt for Stable Diffusion enhancement"""
        return f"""You are a prompt enhancement specialist. Your task is to analyze a user prompt and create a JSON response.

CRITICAL RULES:
1. Return ONLY valid JSON, no other text
2. Use exact keys: Summary, Attributes
3. Attributes must have exactly these keys: Object Class, Environment, Texture
4. Object Class must use objects from: {allowed_objects}
5. All values must be strings
6. No escape sequences inside values
7. Be concise (72 token limit)

Example format:
{{"Summary": "A cardbox on a pallet", "Attributes": {{"Object Class": "cardbox, pallet", "Environment": "Warehouse", "Texture": "Cardboard"}}}}

Remember: ONLY return JSON, nothing else."""

# ==================== JSON VALIDATION ====================
class JSONValidator:
    """Validates and cleans JSON responses"""
    
    @staticmethod
    def clean_response(response_text: str) -> str:
        """Extract and clean JSON from model response"""
        start_idx = response_text.find('{')
        if start_idx == -1:
            return ""
        end_idx = response_text.rfind('}')
        if end_idx == -1:
            return ""
        
        json_str = response_text[start_idx:end_idx + 1]
        
        # Remove only truly problematic control characters (not newlines/tabs)
        json_str = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', json_str)
        
        # Fix common JSON issues
        json_str = re.sub(r',\s*}', '}', json_str)  # Trailing commas in objects
        json_str = re.sub(r',\s*]', ']', json_str)  # Trailing commas in arrays
        json_str = re.sub(r',,+', ',', json_str)      # Multiple commas
        json_str = re.sub(r':\s*,', ': "",', json_str)  # Missing values after colon
        json_str = re.sub(r':\s*}', ': ""}', json_str)  # Missing values before closing brace
        
        return json_str.strip()
    
    @staticmethod
    def validate_structure(
        parsed_json: dict,
        required_attrs: set,
        allowed_objects: set
    ) -> Tuple[bool, str]:
        """Validate JSON structure and content"""
        # Check top-level keys
        required_top = {"Summary", "Attributes"}
        if not isinstance(parsed_json, dict):
            return False, "Response is not a dictionary"
        
        missing_top = required_top - set(parsed_json.keys())
        if missing_top:
            return False, f"Missing top-level keys: {missing_top}"
        
        # Check attributes
        attributes = parsed_json.get("Attributes", {})
        if not isinstance(attributes, dict):
            return False, "Attributes is not a dictionary"
        
        missing_attrs = required_attrs - set(attributes.keys())
        if missing_attrs:
            return False, f"Missing attribute keys: {missing_attrs}"
        
        # Validate object class
        object_class = attributes.get("Object Class", "")
        if not object_class or not object_class.strip():
            return False, "Object Class is empty"
        
        # Check for placeholder text
        placeholders = [
            "main subject", "object type", "surrounding environment",
            "subject/object", "environment and setting"
        ]
        for phrase in placeholders:
            if phrase.lower() in object_class.lower():
                return False, f"Contains placeholder text: '{object_class}'"
        
        # Validate against allowed objects
        objects = [obj.strip() for obj in object_class.split(",")]
        for obj in objects:
            obj_normalized = ObjectNormalizer.normalize(obj)
            if obj_normalized not in allowed_objects:
                return False, f"Object '{obj}' not in allowed list"
        
        # Check all attributes are filled
        for key, value in attributes.items():
            if not value or str(value).strip() == "":
                return False, f"Attribute '{key}' is empty"
        
        return True, "Valid"

# ==================== MAIN ENHANCER CLASS ====================
class PromptEnhancer:
    """Main prompt enhancement engine"""
    
    def __init__(self):
        self.config = EnhancementConfig()
        self.token_manager = TokenManager()
        self.validator = JSONValidator()
    
    def _call_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        num_predict: int
    ) -> str:
        """Call Ollama API"""
        response = ollama.generate(
            model=self.config.MODEL_NAME,
            prompt=user_prompt,
            system=system_prompt,
            stream=False,
            options={
                "temperature": 0.1,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_predict": num_predict,
            }
        )
        return response.get("response", "").strip()
    
    def enhance_for_flux(self, user_prompt: str) -> Optional[Dict]:
        """Enhance prompt for Flux model"""
        allowed_obj_str = ", ".join(sorted(self.config.ALLOWED_OBJECTS))
        system_prompt = SystemPrompts.get_flux_prompt(allowed_obj_str)
        
        full_prompt = f"""Analyze this prompt and create a detailed JSON description for Flux.

User prompt: "{user_prompt}"

ALLOWED OBJECTS: {allowed_obj_str}"""
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                # Call Ollama
                content = self._call_ollama(system_prompt, full_prompt, num_predict=512)
                if not content:
                    print(f"Empty response from Ollama (attempt {attempt + 1})")
                    continue
                
                print(f"Raw response (attempt {attempt + 1}): {content[:200]}...")
                
                # Clean and parse JSON
                json_str = self.validator.clean_response(content)
                if not json_str:
                    print(f"Failed to extract JSON (attempt {attempt + 1})")
                    continue
                
                print(f"Cleaned JSON (attempt {attempt + 1}): {json_str[:200]}...")
                
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    # Try more aggressive fixing
                    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                    json_str = re.sub(r':\s*([^,}\]]*)', lambda m: f': "{m.group(1).strip()}"' if not m.group(1).strip().startswith('"') and m.group(1).strip() else f': {m.group(1)}', json_str)
                    try:
                        parsed = json.loads(json_str)
                    except json.JSONDecodeError as e2:
                        print(f"JSON decode still fails after fixing: {e2}")
                        continue
                
                # Validate structure
                required_attrs = {
                    "Object Class", "Environment", "Lighting",
                    "Texture", "Material", "Distinct Features", "Technical Details"
                }
                is_valid, msg = self.validator.validate_structure(
                    parsed, required_attrs, self.config.ALLOWED_OBJECTS
                )
                
                if not is_valid:
                    print(f"Validation failed (attempt {attempt + 1}): {msg}")
                    continue
                
                # Check token limit
                json_token_count = self.token_manager.count_tokens(json.dumps(parsed))
                if json_token_count > self.config.FLUX_TOKEN_LIMIT:
                    truncated = self.token_manager.truncate_to_limit(
                        json.dumps(parsed),
                        self.config.FLUX_TOKEN_LIMIT
                    )
                    if not truncated.endswith('}'):
                        truncated += '}'
                    try:
                        parsed = json.loads(truncated)
                    except:
                        continue
                
                print(f"✓ Enhancement successful on attempt {attempt + 1}")
                return parsed
                
            except Exception as e:
                print(f"Flux enhancement error (attempt {attempt + 1}): {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return None
    
    def enhance_for_stable_diffusion(self, user_prompt: str) -> Optional[Dict]:
        """Enhance prompt for Stable Diffusion model"""
        allowed_obj_str = ", ".join(sorted(self.config.ALLOWED_OBJECTS))
        system_prompt = SystemPrompts.get_stable_diffusion_prompt(allowed_obj_str)
        
        full_prompt = f"""Analyze this prompt and create a concise JSON description for Stable Diffusion.

User prompt: "{user_prompt}"

ALLOWED OBJECTS: {allowed_obj_str}"""
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                # Call Ollama
                content = self._call_ollama(system_prompt, full_prompt, num_predict=256)
                if not content:
                    continue
                
                # Clean and parse JSON
                json_str = self.validator.clean_response(content)
                if not json_str:
                    continue
                
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError:
                    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                    parsed = json.loads(json_str)
                
                # Validate structure
                required_attrs = {"Object Class", "Environment", "Texture"}
                is_valid, msg = self.validator.validate_structure(
                    parsed, required_attrs, self.config.ALLOWED_OBJECTS
                )
                
                if not is_valid:
                    print(f"Validation failed (attempt {attempt + 1}): {msg}")
                    continue
                
                # Check token limit
                json_token_count = self.token_manager.count_tokens(json.dumps(parsed))
                if json_token_count > self.config.STABLE_DIFFUSION_TOKEN_LIMIT:
                    truncated = self.token_manager.truncate_to_limit(
                        json.dumps(parsed),
                        self.config.STABLE_DIFFUSION_TOKEN_LIMIT
                    )
                    if not truncated.endswith('}'):
                        truncated += '}'
                    try:
                        parsed = json.loads(truncated)
                    except:
                        continue
                
                return parsed
                
            except Exception as e:
                print(f"Stable Diffusion enhancement error (attempt {attempt + 1}): {e}")
                continue
        
        return None
    
    def enhance_prompt(self, user_prompt: str, model_type: str) -> Dict:
        """
        Main enhancement method
        
        Args:
            user_prompt: Original user prompt
            model_type: "flux" or "stable_diffusion"
        
        Returns:
            Dict with enhanced_json and token_count
        """
        if model_type == "flux":
            enhanced = self.enhance_for_flux(user_prompt)
            token_limit = self.config.FLUX_TOKEN_LIMIT
        else:
            enhanced = self.enhance_for_stable_diffusion(user_prompt)
            token_limit = self.config.STABLE_DIFFUSION_TOKEN_LIMIT
        
        if not enhanced:
            raise ValueError(f"Failed to enhance prompt after {self.config.MAX_RETRIES} attempts")
        
        token_count = self.token_manager.count_tokens(json.dumps(enhanced))
        
        return {
            "enhanced_json": enhanced,
            "token_count": token_count,
            "token_limit": token_limit,
            "original_prompt": user_prompt
        }