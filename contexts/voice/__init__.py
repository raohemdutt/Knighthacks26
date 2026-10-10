"""Voice on the box (X16): the Jetson listens through the webcam's microphone and speaks through a Bluetooth speaker.

Everything runs on the box: whisper.cpp transcribes, Piper speaks, and the only network call is to the gateway on
localhost (POST /api/ask, the same path the phone uses). No audio leaves the box and none is kept.
"""
