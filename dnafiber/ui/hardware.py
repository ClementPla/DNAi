import torch
import psutil
import os
import streamlit as st

def get_system_stats():
    stats = {
        "cpu_usage": psutil.cpu_percent(),
        "ram_usage": psutil.virtual_memory().percent,
        "ram_free": psutil.virtual_memory().available / (1024**3), # GB
    }
    
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(gpu_id)
        # Returns (free, total)
        free_vram, total_vram = torch.cuda.mem_get_info()
        
        stats.update({
            "gpu_name": props.name,
            "vram_used": (total_vram - free_vram) / (1024**3),
            "vram_total": total_vram / (1024**3),
            "vram_percent": ((total_vram - free_vram) / total_vram) * 100
        })
    return stats


def sidebar_diagnostics():
    with st.sidebar:
        with st.expander("💻 Hardware Diagnostics", expanded=True):
        
            stats = get_system_stats()
            
            # CPU & RAM
            st.write(f"**CPU Load:** {stats['cpu_usage']}%")
            st.progress(stats['cpu_usage'] / 100)
            
            st.write(f"**System RAM:** {stats['ram_usage']}% ({stats['ram_free']:.1f} GB free)")
            st.progress(stats['ram_usage'] / 100)
            
            # GPU (if available)
            if "gpu_name" in stats:
                st.markdown(f"**GPU:** `{stats['gpu_name']}`")
                vram_text = f"{stats['vram_used']:.1f} / {stats['vram_total']:.1f} GB"
                st.write(f"**VRAM Usage:** {vram_text}")
                st.progress(stats['vram_percent'] / 100)
                
                if stats['vram_percent'] > 85:
                    st.warning("⚠️ High VRAM usage! Consider Sequential mode.")
            else:
                st.info("Running on CPU mode 🐢")
                
                
def create_diagnostics_container():
    with st.sidebar:
        st.divider()
        st.subheader("💻 Hardware Monitor")
        # This is the "hook" we will use to update the UI
        return st.empty() 

def update_diagnostics(container):
    stats = get_system_stats() # The function we wrote earlier
    with container.container():
        # CPU
        st.write(f"**CPU:** {stats['cpu_usage']}%")
        st.progress(stats['cpu_usage'] / 100)
        
        # RAM
        st.write(f"**RAM:** {stats['ram_usage']}%")
        st.progress(stats['ram_usage'] / 100)
        
        # GPU
        if "gpu_name" in stats:
            st.write(f"**VRAM:** {stats['vram_used']:.1f}/{stats['vram_total']:.1f} GB")
            st.progress(stats['vram_percent'] / 100)