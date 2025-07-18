import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.uix.scrollview import ScrollView
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
import os
from datetime import datetime

# Android-specific imports (will be available when built for Android)
try:
    from android.storage import primary_external_storage_path
    from android.permissions import request_permissions, Permission
    ANDROID = True
except ImportError:
    ANDROID = False

# Import your AI function - make sure ai_app3.py is in the same directory
# try:
#     from ai_app3 import main_function
# except ImportError:
#     # Fallback function if ai_app3 is not available
#     def main_function(text):
#         return f"AI processing would happen here with: {text}"

kivy.require('2.0.0')

class CameraApp(App):
    def __init__(self):
        super().__init__()
        self.camera_widget = None
        self.main_layout = None
        self.camera_layout = None
        self.display_layout = None
        self.captured_image_widget = None
        self.text_label = None
        self.input_box = None
        self.message_label = None
        
    def build(self):
        # Request permissions on Android
        if ANDROID:
            self.request_android_permissions()
        
        # Create main layout
        self.main_layout = BoxLayout(orientation='vertical')
        
        # Create camera view layout
        self.create_camera_view()
        
        # Create display view layout
        self.create_display_view()
        
        # Start with camera view
        self.show_camera_view()
        
        return self.main_layout
    
    def request_android_permissions(self):
        """Request necessary permissions on Android"""
        try:
            request_permissions([
                Permission.CAMERA,
                # Permission.WRITE_EXTERNAL_STORAGE,
                # Permission.READ_EXTERNAL_STORAGE
            ])
        except Exception as e:
            Logger.error(f"Permission request failed: {str(e)}")

    def create_camera_view(self):
        """Create the camera view layout"""
        self.camera_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Initialize camera
        self.init_camera(self.camera_layout)

        # Add TextInput for user message
        self.input_box = TextInput(
            hint_text='Enter the type of product here...',
            size_hint=(1, 0.1),
            multiline=False,
            background_color=(0.1, 0.1, 0.1, 1),
            foreground_color=(1, 1, 1, 1),
            cursor_color=(1, 1, 1, 1)
        )
        self.camera_layout.add_widget(self.input_box)

        # Send button
        send_button = Button(
            text='Send',
            size_hint=(1, 0.1),
            background_color=(0.2, 0.6, 1, 1)
        )
        send_button.bind(on_press=self.on_send_button_click)
        self.camera_layout.add_widget(send_button)

        # Status message label
        self.message_label = Label(
            text='Camera ready. Click Send to capture image.',
            size_hint=(1, 0.1),
            color=(1, 1, 1, 1)
        )
        self.camera_layout.add_widget(self.message_label)

    def create_display_view(self):
        """Create the display view layout"""
        self.display_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Create image widget for displaying captured image
        self.captured_image_widget = Image(size_hint=(1, 0.4))
        self.display_layout.add_widget(self.captured_image_widget)
        
        # Create scrollable text area
        self.create_scrollable_text_area()
        
        # Create back button
        back_button = Button(
            text='Back to Camera',
            size_hint=(1, 0.1),
            background_color=(0.8, 0.2, 0.2, 1)
        )
        back_button.bind(on_press=self.on_back_button_click)
        self.display_layout.add_widget(back_button)

    def create_scrollable_text_area(self):
        """Create a scrollable text area"""
        scroll = ScrollView(size_hint=(1, 0.5))

        self.text_label = Label(
            text='',
            size_hint_y=None,
            halign='left',
            valign='top',
            color=(1, 1, 1, 1),
            font_size='16sp'
        )

        # Bind text_size to wrap the text based on label width
        self.text_label.bind(
            width=lambda instance, value: setattr(instance, 'text_size', (value, None))
        )

        # Bind height to texture height so it expands and scrolls
        self.text_label.bind(
            texture_size=lambda instance, value: setattr(instance, 'height', value[1])
        )

        scroll.add_widget(self.text_label)
        self.display_layout.add_widget(scroll)

    def show_display_view(self, image_path):
        """Show the display view with captured image"""
        self.main_layout.clear_widgets()
        self.main_layout.add_widget(self.display_layout)

        # Load and display image
        try:
            self.captured_image_widget.source = image_path
            self.captured_image_widget.reload()
        except Exception as e:
            Logger.error(f"Display: Error loading image - {str(e)}")
            if hasattr(self, 'message_label') and self.message_label:
                self.message_label.text = f"Error loading image: {str(e)}"

        # Get user input from the input box
        user_input = self.input_box.text.strip() if self.input_box else ''

        # Combine image path and user input
        display_text = f"{image_path}\n{user_input}"
        
        # Process with AI function
        # try:
        #     processed_text = main_function(display_text)
        #     self.text_label.text = processed_text
        # except Exception as e:
        #     Logger.error(f"AI processing error: {str(e)}")
        #     self.text_label.text = f"Processing error: {str(e)}\n\nOriginal input: {display_text}"






        file_path = image_path.replace("\\", "/")
        print(file_path)
        import requests
        url = "https://SayanDas123-safety.hf.space/generate_answer"
        files = {'image': open(file_path, "rb")}
        data = {'product': user_input}
        
        try:
            response = requests.post(url, files=files, data=data)
            processed_text = response.text

            import json
            response_dict = json.loads(processed_text)

            # print(response_dict)
            # print(response_dict["answer"])
            self.text_label.text = response_dict["answer"]


            #print(response.json())
        except Exception as E:
            self.text_label.text = f"line 201 Processing error: {str(E)}"



    def show_camera_view(self):
        """Show the camera view"""
        self.main_layout.clear_widgets()
        self.main_layout.add_widget(self.camera_layout)

    def on_back_button_click(self, instance):
        """Handle back button click - return to camera view"""
        if self.input_box:
            self.input_box.text = ''
        self.show_camera_view()
        if self.message_label:
            self.message_label.text = 'Camera ready. Click Send to capture image.'

    def init_camera(self, layout):
        """Initialize camera - Android compatible version"""
        try:
            self.camera_widget = Camera(
                play=True,
                resolution=(640, 480),
                size_hint=(1, 0.8)
            )
            layout.add_widget(self.camera_widget)
            Logger.info("Camera: Using Kivy Camera")
            
        except Exception as e:
            Logger.error(f"Camera: Failed to initialize - {str(e)}")
            # Create placeholder widget
            self.camera_widget = Label(
                text='Camera not available\nPlease check camera permissions',
                size_hint=(1, 0.8),
                color=(1, 0, 0, 1)
            )
            layout.add_widget(self.camera_widget)

    def get_storage_path(self):
        """Get appropriate storage path for the platform"""
        if ANDROID:
            try:
                # Try to get external storage path
                # storage_path = primary_external_storage_path()
                # return os.path.join(storage_path, 'CameraApp')

                #try to get internal storage
                return os.path.join(self.user_data_dir, 'images')
            except Exception as e:
                Logger.warning(f"Could not get external storage: {str(e)}")
                # Fallback to app's internal storage
                return os.path.join(self.user_data_dir, 'images')
        else:
            # Desktop/development environment
            return 'images'

    def on_send_button_click(self, instance):
        """Handle send button click - capture image and show display view"""
        try:
            # Get storage path
            storage_dir = self.get_storage_path()
            
            # Create directory if it doesn't exist
            if not os.path.exists(storage_dir):
                os.makedirs(storage_dir)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(storage_dir, f"captured_{timestamp}.png")
            
            # Capture image using Kivy camera
            if hasattr(self.camera_widget, 'export_to_png'):
                self.camera_widget.export_to_png(filename)
                Logger.info(f"Camera: Image saved as {filename}")
                
                # Switch to display view
                self.show_display_view(filename)
                
            else:
                raise Exception("Camera widget does not support image capture")
            
        except Exception as e:
            Logger.error(f"Camera: Error saving image - {str(e)}")
            if self.message_label:
                self.message_label.text = f"Error: {str(e)}"
                Clock.schedule_once(self.reset_message, 3)

    def reset_message(self, dt):
        """Reset the message label after delay"""
        if self.message_label:
            self.message_label.text = 'Camera ready. Click Send to capture image.'

    def on_stop(self):
        """Cleanup when app closes"""
        Logger.info("App stopping - cleaning up")
        super().on_stop()

if __name__ == '__main__':
    CameraApp().run()