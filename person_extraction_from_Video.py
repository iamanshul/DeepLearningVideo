import sys, argparse, ffmpy, os, datetime, skvideo.io,io, time

from google.cloud import videointelligence
from google.cloud import vision
from PIL import Image, ImageDraw
from google.cloud.vision import types
import numpy as np
global output_log_file
def get_location(path):
    return 'https://storage.googleapis.com/' + path[5:]

def extract_face_image(path, time_stamp):

    filename = create_file_name(path, time_stamp, 'faces/')
    log_string("Timestamp -"+str(time_stamp))
    try:
        input_location = get_location(path)
        ret_value = os.system("ffmpeg -i " + input_location+ " -ss " + str(datetime.timedelta(seconds=time_stamp)) + " -frames:v 1 " + filename)
    except Exception as e: print (e)
    with open(filename,'rb') as img:
        faces = detect_face(img, 4)
        log_string('Found {} face{}'.format(
            len(faces), '' if len(faces) == 1 else 's'))
        img.seek(0) #reset
        output_filename= create_file_name(path, time_stamp,'highlighted/')
        cropped_file_name = highlight_face (img, faces, output_filename, time_stamp)
        log_string ('cropped file name ='+cropped_file_name)
        output_log_file.write("\ntimestamp:"+str(time_stamp)+"\t"+filename+"\t"+output_filename)
        if (cropped_file_name):
            potential_names = get_person_name(cropped_file_name)
            output_log_file.write("\t"+cropped_file_name+"\t"+potential_names[0])

        #write details to a file. timestamp, filename, output_filename, cropped_file_name, name

def get_person_name(filename):
    client = vision.ImageAnnotatorClient()
    if filename.startswith('http') or filename.startswith('gs:'):
        image = types.Image()
        image.source.image_uri = filename
    else:
        with io.open(filename, 'rb') as image_file:
            content = image_file.read()
        image = types.Image(content=content)
    potential_names = client.web_detection(image=image).web_detection
    log_string ('potential name = '+str(potential_names))

    return report(potential_names, 2)

def report(annotations, max_report=5):
    """Prints detected features in the provided web annotations."""
    names =  []
    if annotations.web_entities:
        log_string ('\n{} Web entities found: '.format(
            len(annotations.web_entities)))
        count = 0
        for entity in annotations.web_entities:
            log_string('Score      : {}'.format(entity.score))
            log_string('Description: {}'.format(entity.description))
            names.append(entity.description)
            count += 1
            if count >=max_report:
                break;
    return names
    #return the obvious name




def highlight_face (image, faces, output_filename, time_stamp):
    im = Image.open(image)
    draw = ImageDraw.Draw(im)
    face_boxes = []
    for face in faces:
        box = [(vertex.x, vertex.y)
               for vertex in face.bounding_poly.vertices]
        draw.line(box + [box[0]], width=5, fill='#00ff00')
        im.save(output_filename)
        im.seek(0)
        start_x =box[0][0]
        start_y = box[0][1]
        width = box[1][0] - box[0][0]
        height = box[3][1] - box[0][1]
        start_with_buffer_x = int(start_x - np.ceil(width/2))
        start_with_buffer_y = int(start_y - np.ceil(height/2))
        width_with_buffer = int(start_x + width  + np.ceil(width/2))
        height_with_buffer = int(start_y + height  + np.ceil(height/2))
        crop_box = (start_with_buffer_x, start_with_buffer_y, width_with_buffer, height_with_buffer)
        output_img = im.crop(crop_box)
        cropped_file_name = create_file_name(path,time_stamp,'cropped/')
        output_img.save(cropped_file_name)
        return cropped_file_name


def detect_face(face_file, max_results = 4):
    client = vision.ImageAnnotatorClient()
    content = face_file.read()
    return client.face_detection(image=types.Image(content=content)).face_annotations

def create_file_name(path, time_stamp,folder):
    time_stamp_formatted = str(datetime.timedelta(seconds=time_stamp))
    filename = folder+"person_" + str(time_stamp) +".png"
    count = 1
    while (os.path.exists(filename)):
        filename = folder+"person_" + str(time_stamp) + "_"+str(count)+".png"
        count +=1
    log_string ('created file name:'+filename)
    return filename

def analyze_labels(path):
    """ Detects labels given a GCS path. """
    video_client = videointelligence.VideoIntelligenceServiceClient()
    features = [videointelligence.enums.Feature.LABEL_DETECTION]
    mode = videointelligence.enums.LabelDetectionMode.SHOT_AND_FRAME_MODE
    config = videointelligence.types.LabelDetectionConfig(
        label_detection_mode=mode)
    context = videointelligence.types.VideoContext(
        label_detection_config=config)
    operation = video_client.annotate_video(path, features=features, video_context=context)

    log_string('\nProcessing video for label annotations:')

    # [START check_operation]
    result = operation.result(timeout=90)
    log_string('\nFinished processing.')
    # [END check_operation]
    # [START parse_response]
    frame_labels = result.annotation_results[0].frame_label_annotations
    for i, frame_label in enumerate(frame_labels):
        log_string('Frame label description: {}'.format(
            frame_label.entity.description.encode('utf-8')))
        for category_entity in frame_label.category_entities:
            log_string('\tLabel category description: {}'.format(
                category_entity.description))
            if (category_entity.description in ('person','male', 'female', 'child' )):
                frame = frame_label.frames[0]
                time_offset = (frame.time_offset.seconds +
                    frame.time_offset.nanos / 1e9)
                log_string('\tFirst frame time offset: {}s'.format(time_offset))
                log_string('\tFirst frame confidence: {}'.format(frame.confidence))
                extract_face_image(path, time_offset)
        # [END parse_response]
def log_string(log_statement):
    time_now = datetime.datetime.now().strftime("%H:%M:%S:%f")
    print(time_now + ":-"+log_statement+'\n')


if __name__ == '__main__':
    # [START running_app]
    start_time = time.time()
    movie_to_process = sys.argv[1]
    bucket_name = sys.argv[2]
    path = 'gs://' + bucket_name + '/' + movie_to_process
    logfname = "actor_details_"+str(start_time)
    output_log_file = open(logfname,"w+")
    output_log_file.write("Input filename:"+path+"\n")
    os.system('rm -r faces')
    os.system('rm -r highlighted')
    os.system('rm -r cropped')
    os.system('mkdir faces')
    os.system('mkdir highlighted')
    os.system('mkdir cropped')
    analyze_labels(path)
    log_string ("analysis took", time.time() - start_time, "to run")
    output_log_file.close()
