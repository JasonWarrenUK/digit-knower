import streamlit as st
import base64
import io
from PIL import Image

def canvas_component():
    """A custom Streamlit component for drawing on a canvas."""
    # HTML and JavaScript for the drawing canvas
    canvas_html = """
    <div style="text-align: center;">
        <canvas id="canvas" width="280" height="280" style="border:2px solid #000000; background-color: black;"></canvas>
        <br>
        <button onclick="clearCanvas()" style="margin: 10px;">Clear Canvas</button>
    </div>

    <script>
    var canvas = document.getElementById('canvas');
    var ctx = canvas.getContext('2d');
    var isDrawing = false;
    var lastX = 0;
    var lastY = 0;

    // Set up the canvas
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';

    // Drawing functions
    function draw(e) {
        if (!isDrawing) return;
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
        [lastX, lastY] = [e.offsetX, e.offsetY];
    }

    // Event listeners
    canvas.addEventListener('mousedown', (e) => {
        isDrawing = true;
        [lastX, lastY] = [e.offsetX, e.offsetY];
    });
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', () => {
        isDrawing = false;
        // Send image data to Streamlit
        const imageData = canvas.toDataURL('image/png');
        window.parent.postMessage({
            type: 'streamlit:setComponentValue',
            value: imageData
        }, '*');
    });
    canvas.addEventListener('mouseout', () => isDrawing = false);

    // Touch events
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        isDrawing = true;
        const rect = canvas.getBoundingClientRect();
        [lastX, lastY] = [e.touches[0].clientX - rect.left, e.touches[0].clientY - rect.top];
    });
    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        if (!isDrawing) return;
        const rect = canvas.getBoundingClientRect();
        const x = e.touches[0].clientX - rect.left;
        const y = e.touches[0].clientY - rect.top;
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.stroke();
        [lastX, lastY] = [x, y];
    });
    canvas.addEventListener('touchend', (e) => {
        e.preventDefault();
        isDrawing = false;
        // Send image data to Streamlit
        const imageData = canvas.toDataURL('image/png');
        window.parent.postMessage({
            type: 'streamlit:setComponentValue',
            value: imageData
        }, '*');
    });

    // Clear canvas function
    function clearCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        // Send empty canvas data to Streamlit
        window.parent.postMessage({
            type: 'streamlit:setComponentValue',
            value: null
        }, '*');
    }
    </script>
    """
    # Display the canvas and get the component value
    canvas_data = st.components.v1.html(canvas_html, height=350)
    
    # Store the image data in session state when it changes
    if canvas_data is not None:
        st.session_state.image_data = canvas_data

def get_image_from_canvas():
    """Get the image data from the canvas and convert it to a PIL Image."""
    image_data = st.session_state.get('image_data')
    if image_data and isinstance(image_data, str):
        try:
            # Remove the data URL prefix if present
            if image_data.startswith('data:image/png;base64,'):
                image_data = image_data.split(',')[1]
            # Convert base64 to image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            return image
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None
    return None 