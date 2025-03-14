import json
import os
import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
from models import ChatLLM, WhisperASR, XTTSv2
from home import Home

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# global_device = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = global_device

css = """
#container button.audio-mode-btn, 
#container button.send-text-btn, 
#container button.send-audio-btn, 
#container button.exit-audio-btn {
    padding: 1px 3px !important;
    min-width: 40px !important;
    width: 50px !important;
    font-size: 12px !important;
    height: 35px !important;
    line-height: 16px !important;
    flex-grow: 0 !important;
    flex-shrink: 0 !important;
}

#container #text-mode-container .gr-row {
    display: flex;
    justify-content: space-between; /* Ensures full-width spacing */
    align-items: center;
    width: 100%; /* Ensures it takes the full width */
}

#container #audio-mode-container .gr-row {
    display: flex;
    justify-content: space-between; /* Ensures full-width spacing */
    align-items: center;
    width: 100%; /* Ensures it takes the full width */
}

#container #text-mode-container .gr-textbox {
    flex-grow: 1;
    width: 100%;
}

#container #audio-mode-container .gr-audio {
    flex-grow: 1;
    width: 100%;
}
"""


class Assistant:
    """
    Smart Home Assistant that integrates speech recognition, text generation,
    and text-to-speech to simulate an interactive experience for controlling a
    smart home using natural language.
    """
    
    def __init__(self, 
                 home_name: str = "Casa Interactiva",
                 system_prompt: Optional[str] = None):
        # init smart home
        self.home = Home(home_name=home_name)
        self.use_llm_handler = True
        
        # default system prompt if not provided
        self.system_prompt = self._default_system_prompt()
        if system_prompt:
            self.system_prompt = system_prompt
            
        # init models
        self.init_models()
        
        # create gradio ui
        self.demo = gr.Blocks(title="Smart Home Assistant", css=css)
        self.ui()
        
    def _default_system_prompt(self) -> str:
        return (
            "Eres un asistente de hogar inteligente. "
            "Ayudas a controlar dispositivos, responder preguntas y mantener conversaciones. "
            "\n\n"
            f"Casa: {self.home.state['nombre']}\n"
            f"Habitaciones: {', '.join(self.home.state['habitaciones'].keys())}\n"
            # f"Seguridad: {', '.join(self.home.state['seguridad'].keys())}\n"
        )
        
    def init_models(self):
        # init ASR model
        self.whisper_asr = WhisperASR(model_size="tiny")
        if not self.whisper_asr.is_loaded:
            self.whisper_asr.load()
            
        # init Chat LLM
        self.chat_llm = ChatLLM(model_name="granite")
        if not self.chat_llm.is_loaded:
            self.chat_llm.load()
            
        # init TTS model
        self.tts_model = XTTSv2()
        if not self.tts_model.is_loaded:
            self.tts_model.load()
    
    def ui(self):
        """builds the gradio interface."""
        with self.demo:
            gr.Markdown(f"# {self.home.state['nombre']}")
            gr.Markdown("Interactúa usando la voz o texto.")
            
            with gr.Blocks():
                with gr.Column(variant='compact', elem_id="container"):
                    self.build_assistant()
                    
                    # init the status output with the current home state
                    self.demo.load(
                        fn=lambda: self.home.state,
                        inputs=None,
                        outputs=self.status_output
                    )
    
    def build_assistant(self):
        """build the main assistant interaction."""
        with gr.Column():            
            # chat display and state for conversation history.
            with gr.Row():
                with gr.Column():
                    self.chat_history = gr.Chatbot(elem_id="chatbox", elem_classes="chat-container", label="", type="messages")
                    self.chat_state = gr.State([])
                                    
                    # container for text mode (default)
                    with gr.Column(elem_id="text-mode-container", scale=3) as text_mode:
                        with gr.Row():
                            self.chat_text_input = gr.Textbox(
                                container=False,
                                label="", 
                                placeholder="Text message...", 
                                elem_id="text-input",
                                scale=9
                            )
                        
                            # with gr.Column(scale=1, min_width=10):
                            self.send_text_btn = gr.Button("Enviar", elem_classes="send-text-btn", scale=1, min_width=10)
                            self.audio_mode_btn = gr.Button("Audio", elem_classes="audio-mode-btn", scale=1, min_width=10)
                        
                        self.clear_chat_btn = gr.Button("Limpiar chat", variant="secondary", size="sm", scale=9)
                        
                        with gr.Row():
                            self.use_llm_checkbox = gr.Checkbox(
                                label="Procesar comandos con LLM", 
                                value=True,
                                scale=10,
                                info="Activado: usa LLM para procesar comandos. Desactivado: usa expresiones regulares.",
                                container=False
                            )

                    # container for audio mode (initially hidden)        
                    with gr.Column(elem_id="audio-mode-container", visible=False, scale=3) as audio_mode:
                        with gr.Row():
                            self.audio_recorder = gr.Audio(
                                container=False,
                                sources=["microphone", "upload"],
                                type="numpy", 
                                visible=True, 
                                elem_id="audio-recorder", 
                                label=None,
                                scale=9
                            )
                            # row for the two buttons in audio mode: send and exit.
                            # with gr.Column():
                            self.send_audio_btn = gr.Button("Enviar", elem_classes="send-audio-btn", scale=1, min_width=10)
                            self.exit_audio_btn = gr.Button("Exit", elem_classes="exit-audio-btn", scale=1, min_width=10)
                    
                    # Audio components for TTS
                    with gr.Row():
                        self.ref_audio_file = gr.Audio(
                            label="Reference Voice (Upload a short sample of your voice)", 
                            sources=["upload"], 
                            type="filepath"
                        )
                        self.tts_audio_output = gr.Audio(label="Assistant's Voice Response")
                
                
                with gr.Column():
                    self.system_prompt_input = gr.Textbox(
                        label="System Prompt", 
                        value=self.system_prompt,
                        lines=10
                    )
                    
                    # new row for update prompt button and clear history checkbox
                    with gr.Column():
                        self.update_prompt_btn = gr.Button("Update System Prompt")
                        with gr.Row():
                            self.clear_history_checkbox = gr.Checkbox(
                                label="Limpiar historial del chat", 
                                value=True,
                                info="Activado: elimina el historial del chat al actualizar el prompt del sistema",
                                container=False
                            )   
                        # system prompt update info/warning
                        gr.HTML(
                            "<div style='width: 150%; margin-top: -15px; font-size: 12px; color: #666;'>"
                            "Actualizar el prompt del sistema puede cambiar el comportamiento del asistente. "
                            "Se recomienda limpiar el historial del chat para evitar confusiones."
                            "</div>"
                        )             
                    
                    self.status_output = gr.JSON(label="Current Home Status")
                    self.log_output = gr.Text(label="Log", visible=False)
            
                
            
            # --- bindings ---
            # 1. clicking the Audio button hides text mode and shows audio mode.
            self.audio_mode_btn.click(
                fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
                inputs=[],
                outputs=[text_mode, audio_mode]
            )
            
            # 2. clicking the Exit button in audio mode hides the audio container and returns to text mode.
            self.exit_audio_btn.click(
                fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
                inputs=[],
                outputs=[text_mode, audio_mode]
            )
            
            # 2.1 Update processing method when checkbox changes
            self.use_llm_checkbox.change(
                fn=self._update_processing_method,
                inputs=[self.use_llm_checkbox],
                outputs=[self.log_output]
            )
            
            # 3. sending recorded audio:
            self.send_audio_btn.click(
                fn=self.process_voice_message,
                inputs=[self.audio_recorder, self.ref_audio_file, self.chat_state],
                outputs=[self.chat_history, self.tts_audio_output, self.status_output, self.log_output, self.chat_state, self.chat_text_input]
            )
            
            # 4. sending text messages:
            self.send_text_btn.click(
                fn=self.process_text_message,
                inputs=[self.chat_text_input, self.ref_audio_file, self.chat_state],
                outputs=[self.chat_history, self.tts_audio_output, self.status_output, self.log_output, self.chat_state, self.chat_text_input]
            )
            
            # 5. submitting text messages:
            self.chat_text_input.submit(
                fn=self.process_text_message,
                inputs=[self.chat_text_input, self.ref_audio_file, self.chat_state],
                outputs=[self.chat_history, self.tts_audio_output, self.status_output, self.log_output, self.chat_state, self.chat_text_input]
            )
            
            # 6. updating the system prompt:
            self.update_prompt_btn.click(
                fn=self.update_system_prompt,
                inputs=[self.system_prompt_input, self.clear_history_checkbox, self.chat_state],
                outputs=[self.chat_history, self.log_output, self.chat_state]
            )
            
            # 7. clearing chat history:
            self.clear_chat_btn.click(
                fn=self.clear_chat_history,
                inputs=[],
                outputs=[self.chat_history, self.chat_state, self.log_output]
            )

    def _regex_request_handler(self, user_query: str):
        """
        Handles user requests using regex pattern matching to detect commands.
        Builds a system prompt with contextual information based on detected intents.
        """
        final_prompt = self.system_prompt
                
        # process request to detect intents
        intent_result = self.home.process_request(user_query, None)
        
        # register changes if there was success
        if intent_result.get("success", False):
            num_changes = len(intent_result.get("changes", []))
            self.home.log_event("asistente", "cambios_aplicados", 
                        f"Se han aplicado {num_changes} cambios basados en la solicitud del usuario")
        
        # if intent is not a command or if no intent is detected, don't add additional context
        # is_command is True when a command pattern is detected in the text
        # success is True when at least one change was applied correctly
        if not intent_result.get("is_command", False):
            return final_prompt
        
        # build details of changes only if there are changes detected
        changes_details = []
        if intent_result.get("changes"):
            for change in intent_result["changes"]:
                status = "con éxito" if change.get("success", False) else f"con error: {change.get('reason', 'error desconocido')}"
                changes_details.append(f"Cambio solicitado {change['matched_text']} ({status})")
        
        # build final context based on the results
        context_parts = []
        
        # add information about detected commands
        if changes_details:
            context_parts.append(f"Comandos detectados:")
            context_parts.extend(changes_details)
        
        # build final prompt only if there are context parts
        if context_parts:
            final_prompt = f"{final_prompt}\n\n{chr(10).join(context_parts)}"
        
        return final_prompt

    def _llm_request_handler(self, user_query: str):
        """
        Handles user requests using LLM to detect and process commands.
        This method asks the LLM to analyze if a query contains a command,
        determine if it can be executed, apply changes to the home state,
        and provide a summary of the actions taken.
        
        Returns:
            str: Updated system prompt with context about the changes
        """
        # start with the base system prompt
        final_prompt = self.system_prompt
        
        # convert the current state to a JSON string for the LLM
        home_state_json = json.dumps(self.home.state, indent=3)
        
        # create a prompt that asks the LLM to analyze the query and update the state if needed
        system_prompt_request = f"""
You are an AI assistant that controls a smart home. Analyze the user's message to determine if they're requesting any changes to the home.
Instructions of your task:
1. Determine if the user is requesting a change to the home state (e.g., turning on/off devices, changing temperature, opening/closing doors).
2. If yes, determine if the requested change is possible given the current state and home configuration.
3. If possible, create an updated version of the entire JSON that reflects the changes. Keep all other unchagned values the same as in the current state.
4. Provide a one-line summary of what you did (or why you couldn't do it).

Strictly respond following this exact JSON format:
{{
  "is_command": true/false,
  "is_possible": true/false,
  "updated_state": <instruct>Update the following JSON structure with the values that should be updated. Keep all other values the same as in the current state:</instruct>{home_state_json},
  "summary": "One line summary of changes made or why the change wasn't possible"
}}

If the user is not requesting any changes to the home state, set "is_command" to false.
Ensure that your JSON response is valid and properly formatted. The "updated_state" should strictly keep the exact same structure as the input state.
"""
        
        # ask the LLM to process the request (non-streaming)
        try:
            llm_response = self.chat_llm.generate(user_query, system_prompt=system_prompt_request)
            # we need to remove from the conversation history this system prompt request 
            # to later doesn't include any information about it
            # it need to be only the last user and assistant messages
            self.chat_llm.conv_history = self.chat_llm.conv_history[:-2]
            
            # extract the JSON part from the response
            # the response might contain additional text, so we need to extract the JSON
            import re
            json_match = re.search(r'({.*})', llm_response, re.DOTALL)
            
            if json_match:
                response_data = json.loads(json_match.group(1))
                
                # if the LLM detected a command and it's possible to execute
                if response_data.get("is_command", False) and response_data.get("is_possible", False):
                    # update the home state with the new state
                    if response_data.get("updated_state"):
                        self.home.state = response_data["updated_state"]
                        self.home.log_event("asistente", "cambios_aplicados_llm", 
                                    "Se han aplicado cambios basados en procesamiento LLM")
                    
                    # add the summary to the prompt as context
                    summary = response_data.get("summary", "")
                    if summary:
                        final_prompt = f"{final_prompt}\n\nCambios detectados por LLM:\n{summary}"
            
            # if there was an error processing the JSON or if no command was detected,
            # just return the original prompt
            return final_prompt
                
        except Exception as e:
            # Log the error but don't break the flow - just return the original prompt
            logger.error(f"Error in LLM request handler: {str(e)}")
            return final_prompt

    def process_text_message(
        self, text: str, ref_audio_file: Optional[str], history: List
    ):
        if not text or text.strip() == "":
            yield history, None, self.home.state, "Please enter a command.", history, gr.update(value="")
            return
        
        # initialize or update history
        history = history or []
        history.append({"role": "user", "content": text})
        
        # initial performance to show user message immediately
        yield history, None, self.home.state, "", history, gr.update(value="")
        
        try:            
            # choose which request handler to use
            if self.use_llm_handler:
                # llm-based request handler
                system_prompt = self._llm_request_handler(text)
            else:
                # regex-based request handler
                system_prompt = self._regex_request_handler(text)
            
            # prepare for response
            tts_output = None
            last_response = ""
            
            # generate response with streaming
            for partial_response in self.chat_llm.generate_streaming(text, system_prompt=system_prompt):
                last_response = partial_response
                
                # update history with partial response
                if history and history[-1]["role"] == "assistant":
                    history[-1]["content"] = partial_response
                else:
                    history.append({"role": "assistant", "content": partial_response})
                
                time.sleep(0.05)  # small delay for visual feedback
                yield history, None, self.home.state, "Procesando respuesta...", history, gr.update(value="")
            
            # generate TTS only after completing the response
            if ref_audio_file and last_response:
                tts_output = self._generate_tts(last_response, ref_audio_file)
                yield history, tts_output, self.home.state, "Respuesta generada con audio.", history, gr.update(value="")
            else:
                yield history, None, self.home.state, "Respuesta generada.", history, gr.update(value="")
            
        except Exception as e:
            error_msg = f"Error al procesar la solicitud: {str(e)}"
            logger.error(error_msg)
            
            # add error message to history
            if history and history[-1]["role"] == "assistant":
                history[-1]["content"] = f"Lo siento, ocurrió un problema: {str(e)}"
            else:
                history.append({"role": "assistant", "content": f"Lo siento, ocurrió un problema: {str(e)}"})
            
            yield history, None, self.home.state, error_msg, history, gr.update(value="")
    
    def process_voice_message(
        self, audio, ref_audio_file: Optional[str], history: List
    ):
        """Process a voice command with the same functionality as text messages."""
        if audio is None:
            yield history, None, self.home.state, "No audio provided.", history, gr.update(value="")
            return
        
        try:
            # transcribe the audio using Whisper ASR
            transcription = self.whisper_asr.transcribe(audio)
            
            if not transcription or transcription.strip() == "":
                yield history, None, self.home.state, "Could not transcribe audio clearly.", history, gr.update(value="")
                return
            
            # add transcription to history
            history = history or []
            history.append({"role": "user", "content": f"[Voice] {transcription}"})
            
            # initial performance to show user message immediately
            yield history, None, self.home.state, "Transcribed successfully.", history, gr.update(value="")
            
            # choose request handler
            if self.use_llm_handler:
                # llm-based request handler
                system_prompt = self._llm_request_handler(transcription)
            else:
                # regex-based request handler
                system_prompt = self._regex_request_handler(transcription)
            
            # prepare for response
            tts_output = None
            last_response = ""
            
            # generate response with streaming
            for partial_response in self.chat_llm.generate_streaming(transcription, system_prompt=system_prompt):
                last_response = partial_response
                
                # update history with partial response
                if history and history[-1]["role"] == "assistant":
                    history[-1]["content"] = partial_response
                else:
                    history.append({"role": "assistant", "content": partial_response})
                
                time.sleep(0.05)  # small delay for visual feedback
                yield history, None, self.home.state, "Procesando respuesta...", history, gr.update(value="")
            
            # generate TTS only after completing the response
            if ref_audio_file and last_response:
                tts_output = self._generate_tts(last_response, ref_audio_file)
                yield history, tts_output, self.home.state, "Respuesta generada con audio.", history, gr.update(value="")
            else:
                yield history, None, self.home.state, "Respuesta generada.", history, gr.update(value="")
            
        except Exception as e:
            error_msg = f"Error al procesar comando de voz: {str(e)}"
            logger.error(error_msg)
            
            # add error message to history
            if history and history[-1]["role"] == "assistant":
                history[-1]["content"] = f"Lo siento, ocurrió un problema: {str(e)}"
            else:
                # if history is empty, create one with the error
                if not history:
                    history = [
                        {"role": "user", "content": f"[Voice] (Error en transcripción)"},
                    ]
                history.append({"role": "assistant", "content": f"Lo siento, ocurrió un problema: {str(e)}"})
            
            yield history, None, self.home.state, error_msg, history, gr.update(value="")

    def _generate_tts(self, text: str, ref_audio_file: str) -> Optional[str]:
        """Generate TTS output from text using the reference audio."""
        if not text or not ref_audio_file:
            return None
        
        try:
            # get speaker conditioning from reference audio
            gpt_cond_latent, speaker_embedding = self.tts_model.get_conditioning_latents(ref_audio_file)
            
            # create temporary directory for output if needed
            temp_dir = Path("./temp")
            temp_dir.mkdir(exist_ok=True)
            temp_filename = f"assistant_response_{int(time.time())}.wav"
            save_path = str(temp_dir / temp_filename)
            
            # generate speech
            result = self.tts_model.inference(
                text=text,
                language="es",  # default to Spanish for this application
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                save_path=save_path
            )
            
            return result["file"]
        except Exception as e:
            logger.error(f"TTS generation error: {str(e)}")
            return None
     
    def update_system_prompt(self, new_prompt: str, clear_history: bool, current_history: List) -> Tuple[List, str, List]:
        """
        Update the system prompt for the assistant.
        """
        self.system_prompt = new_prompt
        
        # create message to notify the user about the prompt update
        update_message = {
            "role": "system", 
            "content": "Sistema actualizado: Se ha cambiado el prompt del sistema." + 
                      (" Historial de chat borrado." if clear_history else "")
        }
        
        if clear_history:
            # clear history but add a message notifying about the change
            updated_history = [update_message]
            updated_chat_state = []
            
            # also clear the history in the chat_llm
            if hasattr(self.chat_llm, "conv_history"):
                self.chat_llm.conv_history = []
                
            log_message = "System prompt updated and chat history cleared."
        else:
            # keep history and add a notification message
            updated_history = current_history + [update_message]
            updated_chat_state = current_history
            log_message = "System prompt updated, chat history preserved."
            
        return updated_history, log_message, updated_chat_state

    def clear_chat_history(self) -> Tuple[List, List, str]:
        """
        Clear the chat history and display a system message.
        """
        # create message to notify the user about the cleared chat
        clear_message = {
            "role": "system",
            "content": "🧹 Chat limpiado: Se ha borrado el historial de conversación."
        }
        
        if hasattr(self.chat_llm, "conv_history"):
            self.chat_llm.conv_history = []
        
        # return empty history + notification, empty state, and log message
        return [clear_message], [clear_message], "Chat history cleared successfully."

    def _update_processing_method(self, use_llm: bool) -> str:
        """
        Updates the processing method based on the checkbox value.
        """
        self.use_llm_handler = use_llm
        if use_llm:
            return "Cambio a procesado de peticiones con LLM."
        else:
            return "Cambio a procesado de peticiones con Regex."

    def launch(self, share=True):
        """Wrapper to launch Gradio interface."""
        self.demo.launch(share=share)

# example usage
if __name__ == "__main__":
    assistant = Assistant()
    assistant.launch() 