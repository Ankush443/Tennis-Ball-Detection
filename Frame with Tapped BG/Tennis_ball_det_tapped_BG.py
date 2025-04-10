import cv2
import numpy as np

# Initialize video capture (0 for default camera)
cap = cv2.VideoCapture("Tennis_ball_wall_12.mp4")

# Define the frame size for the physical frame (16:9 aspect ratio)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Define the virtual screen size (16:9 aspect ratio)
VIRTUAL_WIDTH = 1920
VIRTUAL_HEIGHT = 1080

# Create a blank virtual screen with a white background
virtual_screen = np.ones((VIRTUAL_HEIGHT, VIRTUAL_WIDTH, 3), dtype=np.uint8) * 255

# Variables to track ball positions and velocities
prev_position = None
prev_velocity = None
hit_velocity_threshold = 10  # Minimum speed change indicating a hit

# Define color ranges for detecting the taped border
# Adjust these HSV values according to the taped color in your setup
lower_tape_color = np.array([0, 0, 0])  # Example values for red tape
upper_tape_color = np.array([180, 255, 50])  # Example values for red tape

# Function to detect the ball based on color
def detect_ball(frame):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for tennis ball color (adjust these values for better accuracy)
    lower_yellow = np.array([25, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    
    # Create a mask for the color
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assume no ball detected initially
    ball_detected = False
    impact_point = None
    
    # Check for the largest contour
    if contours:
        # Get the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Check if the contour is big enough to be considered a ball
        if cv2.contourArea(largest_contour) > 100:  # Threshold area
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
            if radius > 10:  # Threshold radius
                ball_detected = True
                impact_point = (int(x), int(y))
                # Draw the detected ball on the frame
                cv2.circle(frame, impact_point, int(radius), (0, 255, 0), 2)
                
    return ball_detected, impact_point

# Function to detect the taped border
def detect_border(frame):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a mask for the taped border color
    mask = cv2.inRange(hsv, lower_tape_color, upper_tape_color)
    
    # Find contours of the taped border
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assume no border detected initially
    border_contour = None
    
    if contours:
        # Get the largest contour by area, assuming it is the taped border
        border_contour = max(contours, key=cv2.contourArea)
        # Draw the detected border on the frame (for visualization)
        cv2.drawContours(frame, [border_contour], -1, (255, 0, 0), 2)
    
    return border_contour

# Function to map the physical impact point to the virtual screen
def map_to_virtual_screen(impact_point, frame_width, frame_height, virtual_width, virtual_height):
    # Scaling the impact point to the virtual screen dimensions
    x_ratio = virtual_width / frame_width
    y_ratio = virtual_height / frame_height
    
    virtual_x = int(impact_point[0] * x_ratio)
    virtual_y = int(impact_point[1] * y_ratio)
    
    return (virtual_x, virtual_y)

# Function to determine if the ball has hit the wall based on sudden movement changes
def is_hit(impact_point, prev_position, prev_velocity):
    # Ensure we have a previous position to compare
    if prev_position is None:
        return False
    
    # Calculate the current velocity of the ball
    delta_x = impact_point[0] - prev_position[0]
    delta_y = impact_point[1] - prev_position[1]
    current_velocity = np.sqrt(delta_x**2 + delta_y**2)
    
    # Detect a hit by checking for a sudden drop in velocity (deceleration) or major direction change
    if prev_velocity is not None:
        velocity_change = abs(prev_velocity - current_velocity)
        if velocity_change > hit_velocity_threshold:
            return True
    
    return False

# Function to check if the impact point is inside the detected border
def is_inside_border(impact_point, border_contour):
    if border_contour is None:
        return False
    
    # Use pointPolygonTest to check if the point is inside the contour (-1=outside, 0=on, 1=inside)
    return cv2.pointPolygonTest(border_contour, impact_point, False) > 0

# Main loop for detection and mapping
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect the taped border
    border_contour = detect_border(frame)
    
    # Detect the ball
    ball_detected, impact_point = detect_ball(frame)
    
    if ball_detected and impact_point:
        # Check if the ball hits the wall
        if is_hit(impact_point, prev_position, prev_velocity):
            # Check if the hit is inside the detected border
            if is_inside_border(impact_point, border_contour):
                # Map to virtual screen coordinates
                virtual_point = map_to_virtual_screen(impact_point, FRAME_WIDTH, FRAME_HEIGHT, VIRTUAL_WIDTH, VIRTUAL_HEIGHT)
                
                # Draw a big black dot on the virtual screen at the impact point
                cv2.circle(virtual_screen, virtual_point, 30, (0, 0, 0), -1)  # 30 is the radius of the dot
                
                # Print the impact point coordinates for debugging
                print(f"Impact detected inside border at: {virtual_point}")
            else:
                print("Impact outside the taped border - not considered.")
        
        # Update the previous position and velocity
        if prev_position is not None:
            prev_velocity = np.sqrt((impact_point[0] - prev_position[0])**2 + (impact_point[1] - prev_position[1])**2)
        prev_position = impact_point

    # Resizing the Virtual Screen
    cv2.namedWindow("Virtual Screen", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Virtual Screen", 640, 360)

    # Display the live feed (optional for debugging)
    cv2.imshow('Live Feed', frame)
    
    # Display the virtual screen with impacts
    cv2.imshow('Virtual Screen', virtual_screen)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
