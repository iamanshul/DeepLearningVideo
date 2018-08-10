# DeepLearningVideo
This scripts goes through multiple steps and uses Google Cloud Playform Video APIs to ehance metadata for the video. Overall objective is that this script can take any video in a public location and analyse it. It will then create a file with timestamps when the people appeared. It also does an Image search to find out who the person was and populates that along with the confidence.
Step 1) It extracts the frames with 'faces' in them using VideoIntelligence API
Step 2) Extract scene image from video at the particular timestamp using ffmpeg
Step 3) Extract face image from the frame
step 4) For each of the face images do a Google Image Search to find out the person. 

Please use this and adapt it. Hopefullyt it is of some use. If there are any comments let me know.
