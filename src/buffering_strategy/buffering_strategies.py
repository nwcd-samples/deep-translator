import asyncio
import json
import os
import time
import boto3

from .buffering_strategy_interface import BufferingStrategyInterface


class SilenceAtEndOfChunk(BufferingStrategyInterface):
    """
    A buffering strategy that processes audio at the end of each chunk with
    silence detection.

    This class is responsible for handling audio chunks, detecting silence at
    the end of each chunk, and initiating the transcription process for the
    chunk.

    Attributes:
        client (Client): The templates instance associated with this buffering
                         strategy.
        chunk_length_seconds (float): Length of each audio chunk in seconds.
        chunk_offset_seconds (float): Offset time in seconds to be considered
                                      for processing audio chunks.
    """

    def __init__(self, client, **kwargs):
        """
        Initialize the SilenceAtEndOfChunk buffering strategy.

        Args:
            client (Client): The templates instance associated with this buffering
                             strategy.
            **kwargs: Additional keyword arguments, including
                      'chunk_length_seconds' and 'chunk_offset_seconds'.
        """
        self.client = client
        session = boto3.Session(profile_name='bedrocker')
        self.bedrock_runtime = session.client('bedrock-runtime')
        self.chunk_length_seconds = os.environ.get(
            "BUFFERING_CHUNK_LENGTH_SECONDS"
        )
        if not self.chunk_length_seconds:
            self.chunk_length_seconds = kwargs.get("chunk_length_seconds")
        self.chunk_length_seconds = float(self.chunk_length_seconds)

        self.chunk_offset_seconds = os.environ.get(
            "BUFFERING_CHUNK_OFFSET_SECONDS"
        )
        if not self.chunk_offset_seconds:
            self.chunk_offset_seconds = kwargs.get("chunk_offset_seconds")
        self.chunk_offset_seconds = float(self.chunk_offset_seconds)

        self.error_if_not_realtime = os.environ.get("ERROR_IF_NOT_REALTIME")
        if not self.error_if_not_realtime:
            self.error_if_not_realtime = kwargs.get(
                "error_if_not_realtime", False
            )

        self.processing_flag = False

    def process_audio(self, websocket, vad_pipeline, asr_pipeline):
        """
        Process audio chunks by checking their length and scheduling
        asynchronous processing.

        This method checks if the length of the audio buffer exceeds the chunk
        length and, if so, it schedules asynchronous processing of the audio.

        Args:
            websocket: The WebSocket connection for sending transcriptions.
            vad_pipeline: The voice activity detection pipeline.
            asr_pipeline: The automatic speech recognition pipeline.
        """
        chunk_length_in_bytes = (
            self.chunk_length_seconds
            * self.client.sampling_rate
            * self.client.samples_width
        )
        if len(self.client.buffer) > chunk_length_in_bytes:
            if self.processing_flag:
                exit(
                    "Error in realtime processing: tried processing a new "
                    "chunk while the previous one was still being processed"
                )

            self.client.scratch_buffer += self.client.buffer
            self.client.buffer.clear()
            self.processing_flag = True
            # Schedule the processing in a separate task
            asyncio.create_task(
                self.process_audio_async(websocket, vad_pipeline, asr_pipeline)
            )

    async def translate(self, websocket, transcript):

        claude_response = ''
        conversation_history = []
        system_message = f'You are good at translating English to Chinese. ' \
                         f'Please translate below sentences to Chinese directly and formally' \
                         f'Constraints' \
                         f"1. Don't insert any thinking process into the response" \
                         f"2. Don't place any prefix before the response" \
                         f"3. Don't place any subfix after the response"
        prompty_prefix = "Please translate below sentences to Chinese directly without any prefilling \n"
        # validated_history = [
        #     msg for msg in conversation_history if msg.get("content").strip()]
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

        # conversation_history.append(
        #     {"role": "user", "content": f"{prompty_prefix}{transcript}"})
        conversation_history.append(
            {"role": "user", "content": f"{transcript}"})
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": conversation_history,
            "system": system_message,
        })

        response = self.bedrock_runtime.invoke_model_with_response_stream(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json",
        )
        # print(response)
        # logger.info(f"Claude process completed, begin to resolve Claude response")
        stream = response["body"]
        if stream:
            for event in stream:
                # Mapping to data part
                # SSE Response Sample
                # event: content_block_delta
                # data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}
                chunk = json.loads(event["chunk"]["bytes"])
                if chunk:
                    if chunk['type'] == 'content_block_delta':
                        text = chunk['delta']['text']
                        # print(Fore.CYAN + text + Style.RESET_ALL, end="", flush=True)
                        claude_response += text

                        # status_details=(f"{status_details}\n{claude_response}")
            json_result = json.dumps({"llm_type":"claude3", "text": claude_response })
            await websocket.send(json_result)
            # global status_details
            # status_details = f"{claude_response}"
            # st.write(status_details)

    async def process_audio_async(self, websocket, vad_pipeline, asr_pipeline):
        """
        Asynchronously process audio for activity detection and transcription.

        This method performs heavy processing, including voice activity
        detection and transcription of the audio data. It sends the
        transcription results through the WebSocket connection.

        Args:
            websocket (Websocket): The WebSocket connection for sending
                                   transcriptions.
            vad_pipeline: The voice activity detection pipeline.
            asr_pipeline: The automatic speech recognition pipeline.
        """
        start = time.time()
        vad_results = await vad_pipeline.detect_activity(self.client)

        if len(vad_results) == 0:
            self.client.scratch_buffer.clear()
            self.client.buffer.clear()
            self.processing_flag = False
            return

        last_segment_should_end_before = (
            len(self.client.scratch_buffer)
            / (self.client.sampling_rate * self.client.samples_width)
        ) - self.chunk_offset_seconds
        if vad_results[-1]["end"] < last_segment_should_end_before:
            transcription = await asr_pipeline.transcribe(self.client)
            if transcription["language"] != 'zh' and transcription["text"] != "":
                # print(transcription["text"])
                end = time.time()
                transcription["processing_time"] = end - start
                json_transcription = json.dumps(transcription)
                await websocket.send(json_transcription)
                await self.translate(websocket, transcription["text"])

            self.client.scratch_buffer.clear()
            self.client.increment_file_counter()

        self.processing_flag = False
