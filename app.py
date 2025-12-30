"""
Multimodal AI Pipeline Application
===================================
Integrates: LLM Prompt Enhancement (Ollama) → Diffusion Image Generation (Flux/SD) → 
Vision-Language Annotation (Grounding DINO)
"""

import streamlit as st
import os
from pathlib import Path
from PIL import Image
import json
from datetime import datetime

# Import custom modules
from modules.prompt_enhancement_module import PromptEnhancer
from modules.image_generation_module import ImageGenerator
from modules.annotation_module import AnnotationEngine

# ==================== CONFIGURATION ====================
OUTPUT_BASE_DIR = "pipeline_outputs"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# ==================== SESSION STATE INITIALIZATION ====================
def initialize_session_state():
    """Initialize all session state variables"""
    if 'stage' not in st.session_state:
        st.session_state.stage = 1
    if 'user_prompt' not in st.session_state:
        st.session_state.user_prompt = ""
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "Flux"
    if 'enhanced_prompts' not in st.session_state:
        st.session_state.enhanced_prompts = {}
    if 'selected_prompt_type' not in st.session_state:
        st.session_state.selected_prompt_type = "enhanced"
    if 'generated_images' not in st.session_state:
        st.session_state.generated_images = []
    if 'annotations' not in st.session_state:
        st.session_state.annotations = {}

# ==================== STAGE NAVIGATION ====================
def advance_stage(target_stage):
    """Advance to next stage"""
    st.session_state.stage = target_stage
    st.rerun()

def reset_pipeline():
    """Reset entire pipeline"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# ==================== STAGE 1: USER INPUT ====================
def render_stage1_input():
    """Stage 1: User Input & Model Selection"""
    st.header("🎯 Stage 1: Input Configuration")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_prompt = st.text_area(
            "Enter your image generation prompt:",
            value=st.session_state.user_prompt,
            height=120,
            placeholder="e.g., A robotic arm picking up a cardboard box in an industrial warehouse with bright overhead lighting",
            key="prompt_input"
        )
    
    with col2:
        st.markdown("### Model Selection")
        model = st.selectbox(
            "Generation Model:",
            ["Flux", "Stable Diffusion"],
            index=0 if st.session_state.selected_model == "Flux" else 1,
            key="model_select"
        )
        
        flux_info = "• Higher quality\n• 512 token limit"
        sd_info = "• Faster generation\n• 72 token limit"
        info_text = flux_info if model == 'Flux' else sd_info
        st.info(f"**{model}**\n\n{info_text}")
    
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("🚀 Enhance Prompt", use_container_width=True, type="primary"):
            if user_prompt.strip():
                st.session_state.user_prompt = user_prompt
                st.session_state.selected_model = model
                advance_stage(2)
            else:
                st.error("⚠️ Please enter a valid prompt")

# ==================== STAGE 2: PROMPT ENHANCEMENT ====================
def render_stage2_enhancement():
    """Stage 2: Prompt Enhancement"""
    st.header("🧠 Stage 2: Prompt Enhancement")
    
    # Display original prompt
    with st.expander("📝 Original Prompt", expanded=False):
        st.write(st.session_state.user_prompt)
    
    # Initialize enhancer
    if not st.session_state.enhanced_prompts:
        with st.spinner(f"🔄 Enhancing prompt for {st.session_state.selected_model}..."):
            try:
                enhancer = PromptEnhancer()
                model_type = "flux" if st.session_state.selected_model == "Flux" else "stable_diffusion"
                enhanced_data = enhancer.enhance_prompt(
                    st.session_state.user_prompt,
                    model_type
                )
                st.session_state.enhanced_prompts[model_type] = enhanced_data
            except Exception as e:
                st.error(f"❌ Enhancement failed: {str(e)}")
                if st.button("← Back to Input"):
                    advance_stage(1)
                return
    
    # Display enhanced prompt
    model_type = "flux" if st.session_state.selected_model == "Flux" else "stable_diffusion"
    enhanced_data = st.session_state.enhanced_prompts.get(model_type, {})
    
    if enhanced_data:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### ✨ Enhanced Prompt")
            enhanced_json = enhanced_data.get('enhanced_json', {})
            
            # Display summary
            summary = enhanced_json.get('Summary', 'N/A')
            st.info(f"**Summary:** {summary}")
            
            # Display attributes
            attributes = enhanced_json.get('Attributes', {})
            if attributes:
                st.markdown("**Detailed Attributes:**")
                for key, value in attributes.items():
                    st.markdown(f"- **{key}:** {value}")
        
        with col2:
            st.markdown("### 📊 Token Analysis")
            token_count = enhanced_data.get('token_count', 0)
            token_limit = 512 if model_type == "flux" else 72
            
            token_percentage = (token_count / token_limit) * 100
            
            st.metric("Token Count", f"{token_count} / {token_limit}")
            st.progress(min(token_percentage / 100, 1.0))
            
            if token_count > token_limit:
                st.error("⚠️ Exceeds limit!")
            elif token_count > token_limit * 0.9:
                st.warning("⚠️ Near limit")
            else:
                st.success("✓ Within limit")
    
    st.markdown("---")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            advance_stage(1)
    with col3:
        if st.button("Continue →", use_container_width=True, type="primary"):
            advance_stage(3)

# ==================== STAGE 3: IMAGE GENERATION ====================
def render_stage3_generation():
    """Stage 3: Image Generation Configuration"""
    st.header("🎨 Stage 3: Image Generation")
    
    # Prompt selection
    st.markdown("### 📝 Select Prompt Type")
    prompt_type = st.radio(
        "Choose prompt to use:",
        ["enhanced", "original"],
        format_func=lambda x: "✨ Enhanced Prompt (Recommended)" if x == "enhanced" else "📄 Original Prompt",
        horizontal=True,
        key="prompt_type_radio"
    )
    st.session_state.selected_prompt_type = prompt_type
    
    # Display selected prompt
    if prompt_type == "original":
        st.info(f"**Selected:** {st.session_state.user_prompt}")
    else:
        model_type = "flux" if st.session_state.selected_model == "Flux" else "stable_diffusion"
        enhanced_data = st.session_state.enhanced_prompts.get(model_type, {})
        summary = enhanced_data.get('enhanced_json', {}).get('Summary', st.session_state.user_prompt)
        st.info(f"**Selected:** {summary}")
    
    st.markdown("---")
    
    # Generation parameters
    st.markdown("### ⚙️ Generation Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model = st.selectbox(
            "Generation Model:",
            ["Flux", "Stable Diffusion"],
            index=0 if st.session_state.selected_model == "Flux" else 1
        )
        
        aspect_ratio = st.selectbox(
            "Aspect Ratio:",
            ["1:1 (1024×1024)", "16:9 (1024×576)", "4:3 (800×600)", "2:1 (1024×512)"]
        )
    
    with col2:
        num_images = st.slider(
            "Number of Images:",
            min_value=1,
            max_value=10,
            value=1
        )
        
        image_format = st.selectbox(
            "Output Format:",
            ["PNG", "JPEG"]
        )
    
    st.markdown("---")
    
    # Generate button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            advance_stage(2)
    with col3:
        if st.button("🎨 Generate", use_container_width=True, type="primary"):
            with st.spinner("🎨 Generating images..."):
                try:
                    # Prepare prompt
                    if prompt_type == "original":
                        final_prompt = st.session_state.user_prompt
                    else:
                        model_type = "flux" if model == "Flux" else "stable_diffusion"
                        enhanced_data = st.session_state.enhanced_prompts.get(model_type, {})
                        enhanced_json = enhanced_data.get('enhanced_json', {})
                        
                        # Construct detailed prompt from JSON
                        summary = enhanced_json.get('Summary', '')
                        attributes = enhanced_json.get('Attributes', {})
                        attr_text = ", ".join([f"{k}: {v}" for k, v in attributes.items()])
                        final_prompt = f"{summary}. {attr_text}"
                    
                    # Initialize generator
                    generator = ImageGenerator()
                    
                    # Parse aspect ratio
                    aspect_map = {
                        "1:1 (1024×1024)": "1:1",
                        "16:9 (1024×576)": "16:9",
                        "4:3 (800×600)": "4:3",
                        "2:1 (1024×512)": "2:1"
                    }
                    aspect = aspect_map.get(aspect_ratio, "1:1")
                    
                    # Generate images
                    image_paths = generator.generate_images(
                        prompt=final_prompt,
                        num_images=num_images,
                        aspect_ratio=aspect,
                        model_name=model,
                        image_format=image_format
                    )
                    
                    st.session_state.generated_images = image_paths
                    st.success(f"✅ Generated {len(image_paths)} image(s)")
                    advance_stage(4)
                    
                except Exception as e:
                    st.error(f"❌ Generation failed: {str(e)}")

# ==================== STAGE 4: PREVIEW ====================
def render_stage4_preview():
    """Stage 4: Image Preview"""
    st.header("🖼️ Stage 4: Generated Images Preview")
    
    if not st.session_state.generated_images:
        st.warning("No images generated yet")
        if st.button("← Back to Generation"):
            advance_stage(3)
        return
    
    # Display images in grid
    num_cols = min(3, len(st.session_state.generated_images))
    cols = st.columns(num_cols)
    
    for idx, img_path in enumerate(st.session_state.generated_images):
        with cols[idx % num_cols]:
            try:
                image = Image.open(img_path)
                st.image(image, caption=f"Image {idx + 1}", use_container_width=True)
                
                # Download button
                with open(img_path, "rb") as file:
                    st.download_button(
                        label=f"⬇️ Download {idx + 1}",
                        data=file,
                        file_name=os.path.basename(img_path),
                        mime=f"image/{'png' if img_path.endswith('.png') else 'jpeg'}",
                        key=f"download_{idx}"
                    )
            except Exception as e:
                st.error(f"Failed to load image: {e}")
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Regenerate", use_container_width=True):
            advance_stage(3)
    with col3:
        if st.button("Annotate →", use_container_width=True, type="primary"):
            advance_stage(5)

# ==================== STAGE 5: ANNOTATION ====================
def render_stage5_annotation():
    """Stage 5: Annotation Generation"""
    st.header("🎯 Stage 5: Annotation Generation")
    
    if not st.session_state.generated_images:
        st.warning("No images to annotate")
        return
    
    # Annotation mode selection
    st.markdown("### 🔧 Annotation Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        annotation_mode = st.selectbox(
            "Annotation Mode:",
            ["grounding_dino", "blip2", "basic"],
            format_func=lambda x: {
                "grounding_dino": "🎯 Grounding DINO (Object Detection)",
                "blip2": "💬 BLIP-2 (Image Captioning)",
                "basic": "📦 Basic (Bounding Box Only)"
            }[x]
        )
    
    with col2:
        if annotation_mode == "grounding_dino":
            detection_prompt = st.text_input(
                "Detection Prompt:",
                value="cardbox, bottle, traffic cone, robotic arm",
                help="Comma-separated list of objects to detect"
            )
        else:
            detection_prompt = ""
    
    st.markdown("---")
    
    # Generate annotations
    if st.button("🎯 Generate Annotations", use_container_width=True, type="primary"):
        with st.spinner("🔄 Processing annotations..."):
            try:
                annotator = AnnotationEngine()
                
                for img_path in st.session_state.generated_images:
                    result = annotator.annotate_image(
                        image_path=img_path,
                        mode=annotation_mode,
                        prompt=detection_prompt if annotation_mode == "grounding_dino" else None
                    )
                    
                    st.session_state.annotations[img_path] = result
                
                st.success("✅ Annotations generated successfully")
                
            except Exception as e:
                st.error(f"❌ Annotation failed: {str(e)}")
                return
    
    # Display annotated images
    if st.session_state.annotations:
        st.markdown("### 📊 Annotated Results")
        
        for img_path, annotation_data in st.session_state.annotations.items():
            st.markdown(f"#### {os.path.basename(img_path)}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if 'annotated_image_path' in annotation_data:
                    try:
                        annotated_img = Image.open(annotation_data['annotated_image_path'])
                        st.image(annotated_img, caption="Annotated Image", use_container_width=True)
                    except:
                        st.error("Failed to load annotated image")
            
            with col2:
                st.markdown("**Detection Summary:**")
                detections = annotation_data.get('detections', [])
                st.metric("Objects Detected", len(detections))
                
                if detections:
                    st.markdown("**Detected Objects:**")
                    for det in detections[:5]:  # Show first 5
                        st.markdown(f"- {det.get('class_name', 'Unknown')}: {det.get('confidence', 0):.2f}")
                
                # Download annotation file
                if 'annotation_file_path' in annotation_data:
                    with open(annotation_data['annotation_file_path'], 'r') as f:
                        st.download_button(
                            label="⬇️ Download Annotation",
                            data=f.read(),
                            file_name=os.path.basename(annotation_data['annotation_file_path']),
                            mime="text/plain"
                        )
            
            st.markdown("---")
    
    # Final navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back to Preview", use_container_width=True):
            advance_stage(4)
    with col3:
        if st.button("🔄 New Pipeline", use_container_width=True, type="primary"):
            reset_pipeline()

# ==================== MAIN APPLICATION ====================
def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="Multimodal AI Pipeline",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 Multimodal AI Pipeline")
        st.markdown("---")
        
        # Stage indicator
        st.markdown("### 📍 Pipeline Progress")
        stages = [
            ("1️⃣", "Input", 1),
            ("2️⃣", "Enhancement", 2),
            ("3️⃣", "Generation", 3),
            ("4️⃣", "Preview", 4),
            ("5️⃣", "Annotation", 5)
        ]
        
        for icon, name, stage_num in stages:
            if st.session_state.stage == stage_num:
                st.markdown(f"**{icon} {name}** ◀")
            elif st.session_state.stage > stage_num:
                st.markdown(f"~~{icon} {name}~~ ✓")
            else:
                st.markdown(f"{icon} {name}")
        
        st.markdown("---")
        
        # System info
        st.markdown("### ℹ️ System Info")
        st.markdown(f"**Current Stage:** {st.session_state.stage}/5")
        if st.session_state.selected_model:
            st.markdown(f"**Model:** {st.session_state.selected_model}")
        
        st.markdown("---")
        
        if st.button("🔄 Reset Pipeline", use_container_width=True):
            reset_pipeline()
    
    # Main content area - render current stage
    if st.session_state.stage == 1:
        render_stage1_input()
    elif st.session_state.stage == 2:
        render_stage2_enhancement()
    elif st.session_state.stage == 3:
        render_stage3_generation()
    elif st.session_state.stage == 4:
        render_stage4_preview()
    elif st.session_state.stage == 5:
        render_stage5_annotation()

if __name__ == "__main__":
    main()