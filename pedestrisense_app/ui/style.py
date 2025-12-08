# pedestrisense_app/ui/style.py
APP_STYLE = """
QWidget {
    background-color: #0d0f14;
    color: white;
    font-family: 'Segoe UI';
}

#Title {
    font-size: 40px;
    font-weight: 800;
    color: #7cf3ff;
    margin-bottom: -10px;
}

#Subtitle {
    font-size: 20px;
    font-weight: 500;
    color: #c8f7ff;
}

.GlassCard {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 18px;
    border: 1px solid rgba(0, 255, 255, 0.35);
    padding: 10px;
    backdrop-filter: blur(15px);
}

.GlassCard QLabel {
    color: #b3f9ff;
    font-size: 20px;
    font-weight: 600;
}

.VideoFrame {
    border-radius: 10px;
    border: 2px solid rgba(0, 255, 255, 0.5);
}
"""
