import json
import os
import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import torch

from models import ChatLLM, WhisperASR, VITTS
from home import Home

# configure logging
logging.basicConfig(level=logging.INFO)


class Assistant:
    """
    smart home assistant that integrates speech recognition, text generation,
    and text-to-speech to control a smart home using natural language
    """
    
    def __init__(self, 
                 home_name: str = "Casa Interactiva",
                 system_prompt: Optional[str] = None):
        # init smart home
        self.home = Home(home_name=home_name)
        self.use_llm_handler = True
        
        # init logger
        self.logger = logging.getLogger(__name__)
        
        # init tracking TTS language selection
        self.current_tts_lang = "spa"
        
        # default system prompt if not provided
        self.system_prompt = self._default_system_prompt()
        if system_prompt:
            self.system_prompt = system_prompt
            
        # init models
        self.init_models()
        
        # create gradio ui
        self.demo = gr.Blocks(title="Smart Home Assistant")
        self.ui()
        
    def _default_system_prompt(self) -> str:
        return (
            "Eres un asistente de hogar inteligente. "
            "Ayudas a controlar dispositivos, responder preguntas y mantener conversaciones. "
            "\n\n"
            f"Casa: {self.home.state['nombre']}\n"
            f"Habitaciones: {', '.join(self.home.state['habitaciones'].keys())}\n"
        )
        
    def init_models(self):
        """initialize AI models"""
        # init ASR model
        self.whisper_asr = WhisperASR(model_size="tiny")
        if not self.whisper_asr.is_loaded:
            self.whisper_asr.load()
            
        # init Chat LLM
        self.chat_llm = ChatLLM(model_name="llama3.2-3b")
        if not self.chat_llm.is_loaded:
            self.chat_llm.load()
            
        # init TTS model with Spanish as default
        self.tts_model = VITTS(language=self.current_tts_lang)
        if not self.tts_model.is_loaded:
            self.tts_model.load()
    
    def ui(self):
        """build the gradio interface"""
        with self.demo:
            gr.Markdown(f"# {self.home.state['nombre']}")
            gr.Markdown("Interactúa usando la voz o texto.")
            
            with gr.Blocks():
                with gr.Column(variant='compact', elem_id="container"):
                    # build the assistant block
                    self.build_assistant_block()
                    
                    # init the status output with the current home state
                    self.demo.load(
                        fn=lambda: self.home.state,
                        inputs=None,
                        outputs=self.status_output
                    )
    
    def build_assistant_block(self):
        """build the main assistant interaction block"""
        with gr.Column():            
            # chat display and state for conversation history
            with gr.Row():
                # chat history and input
                with gr.Column(elem_id="chat-container"):
                    self.chat_history = gr.Chatbot(
                        elem_id="chatbox", 
                        elem_classes="chat-container", 
                        label="", 
                        type="messages"
                    )
                    self.chat_state = gr.State([])
                                    
                    # container for text mode (default)
                    with gr.Column(elem_classes="text-mode-container") as text_mode:
                        with gr.Row():
                            self.chat_text_input = gr.Textbox(
                                container=False,
                                label="", 
                                placeholder="Escribe un mensaje...", 
                                elem_id="text-input",
                                elem_classes="text-input-chat"
                            )
                        
                        with gr.Row():
                            self.send_text_btn = gr.Button("Enviar", elem_classes="send-text-btn")
                            self.audio_mode_btn = gr.Button("Audio", elem_classes="audio-mode-btn")

                    # container for audio mode (hidden initially)        
                    with gr.Column(elem_id="audio-mode-container", visible=False) as audio_mode:
                        with gr.Row():
                            self.audio_recorder = gr.Audio(
                                container=False,
                                sources=["microphone", "upload"],
                                type="numpy", 
                                visible=True, 
                                elem_id="audio-recorder", 
                                label=None,
                            )

                        with gr.Row():
                            # row for the two buttons in audio mode: send and exit
                            self.send_audio_btn = gr.Button("Enviar", elem_classes="send-audio-btn")
                            self.exit_audio_btn = gr.Button("Salir", elem_classes="exit-audio-btn")
                    
                    # clear chat button
                    with gr.Row():
                        self.clear_chat_btn = gr.Button("Limpiar chat", variant="secondary", size="sm")
                    
                    # checkbox for LLM processing
                    with gr.Row():
                        self.use_llm_checkbox = gr.Checkbox(
                            label="Procesar comandos con LLM", 
                            value=True,
                            info="Activado: usa LLM para procesar comandos. Desactivado: usa expresiones regulares. Se bloquea en ON para idiomas distintos al español.",
                            container=False,
                            # only interactive if the language is Spanish
                            interactive=(self.current_tts_lang == "spa")
                        )

                    # LLM model dropdown
                    with gr.Row():
                        self.llm_model_dropdown = gr.Dropdown(
                            choices=list(ChatLLM.models.keys()),
                            interactive=True,
                            container=False,
                            value="llama3.2-3b",
                            label="Modelo LLM",
                            info="Selecciona el modelo de lenguaje a utilizar",
                        )
                    
                    # audio components for TTS
                    self.tts_audio_output = gr.Audio(
                        label="Respuesta de voz del asistente",
                        type="numpy"
                    )

                    # TTS configuration
                    with gr.Row():                        
                        with gr.Accordion("Configuración de TTS", open=False):
                            self.tts_language = gr.Dropdown(
                                choices=[
                                    ("Español", "spa"), 
                                    ("English", "eng")
                                ],
                                interactive=True,
                                container=False,
                                label="Idioma de la respuesta de voz",
                                value="spa"
                            )   
                
                # system prompt input and status output
                with gr.Column():
                    self.system_prompt_input = gr.Textbox(
                        label="Prompt del sistema", 
                        value=self.system_prompt,
                        lines=10
                    )
                    
                    # update prompt button and clear history checkbox
                    with gr.Column():
                        self.update_prompt_btn = gr.Button("Actualizar prompt del sistema", size="sm")
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
                    
                    # home status output
                    self.status_output = gr.JSON(
                        label="Estado actual del hogar",
                        height=417
                    )
                    
                    # log output
                    self.log_output = gr.Text(
                        label="Log", 
                        visible=True, 
                        lines=2,
                        container=False, 
                        elem_classes="log-output"
                    )
            
            # --- bindings ---
            # 1. clicking the Audio button hides text mode and shows audio mode
            self.audio_mode_btn.click(
                fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
                inputs=[],
                outputs=[text_mode, audio_mode]
            )
            
            # 2. clicking the Exit button in audio mode returns to text mode
            self.exit_audio_btn.click(
                fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
                inputs=[],
                outputs=[text_mode, audio_mode]
            )
            
            # 2.1 update processing method when checkbox changes
            self.use_llm_checkbox.change(
                fn=self._update_processing_method,
                inputs=[self.use_llm_checkbox],
                outputs=[self.log_output]
            )
            
            # 3. sending recorded audio
            self.send_audio_btn.click(
                fn=self.process_voice_message,
                inputs=[self.audio_recorder, self.chat_state],
                outputs=[self.chat_history, self.tts_audio_output, self.status_output, self.log_output, self.chat_state, self.chat_text_input]
            )
            
            # 4. sending text messages
            self.send_text_btn.click(
                fn=self.process_text_message,
                inputs=[self.chat_text_input, self.chat_state],
                outputs=[self.chat_history, self.tts_audio_output, self.status_output, self.log_output, self.chat_state, self.chat_text_input]
            )
            
            # 5. submitting text messages
            self.chat_text_input.submit(
                fn=self.process_text_message,
                inputs=[self.chat_text_input, self.chat_state],
                outputs=[self.chat_history, self.tts_audio_output, self.status_output, self.log_output, self.chat_state, self.chat_text_input]
            )
            
            # 6. updating the system prompt
            self.update_prompt_btn.click(
                fn=self.update_system_prompt,
                inputs=[self.system_prompt_input, self.clear_history_checkbox, self.chat_state],
                outputs=[self.chat_history, self.log_output, self.chat_state]
            )
            
            # 7. clearing chat history
            self.clear_chat_btn.click(
                fn=self.clear_chat_history,
                inputs=[],
                outputs=[self.chat_history, self.chat_state, self.log_output]
            )

            # 8. add model switching binding
            self.llm_model_dropdown.change(
                fn=self._switch_llm_model,
                inputs=[self.llm_model_dropdown],
                outputs=[self.log_output]
            )
            
            # 9. add TTS language change handler
            self.tts_language.change(
                fn=self._switch_tts_language,
                inputs=[self.tts_language],
                outputs=[self.log_output, self.use_llm_checkbox]
            )

    def _regex_request_handler(self, user_query: str):
        """handle user requests using regex pattern matching"""
        final_prompt = self.system_prompt
                
        # process request to detect intents
        intent_result = self.home.process_request(user_query, None)
        
        home_state_json = json.dumps(self.home.state, indent=3)
        
        # register changes if there was success
        if intent_result.get("success", False):
            num_changes = len(intent_result.get("changes", []))
            self.home.log_event("asistente", "cambios_aplicados", 
                        f"Se han aplicado {num_changes} cambios basados en la solicitud del usuario")
        
        # if intent is not a command or if no intent is detected, don't add additional context
        if not intent_result.get("is_command", False):
            final_prompt = f"{final_prompt}\nEstado actual del hogar:\n{home_state_json}"
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
        """handle user requests using LLM to detect and process commands"""
        final_prompt = self.system_prompt
        home_state_json = json.dumps(self.home.state, indent=3)
        
        # create a prompt that asks the LLM to analyze the query and update the state if needed
        system_prompt_request = f"""
You are an helpful assistant that controls a smart home. Analyze the user's message to determine if they're requesting any changes to the home.
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
  "summary": "Provide a concise, one-line summary clearly stating the outcome of the user's requested command or directly addressing their query (e.g. about the home's current state). "
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
            
            return final_prompt
                
        except Exception as e:
            self.logger.error(f"Error in LLM request handler: {str(e)}")
            return final_prompt

    def process_text_message(self, text: str, history: List):
        """process a text message and generate response"""
        if not text or text.strip() == "":
            yield history, None, self.home.state, "Por favor ingresa un comando.", history, gr.update(value="")
            return
        
        # initialize or update history
        history = history or []
        history.append({"role": "user", "content": text})
        
        # initial performance to show user message immediately
        yield history, None, self.home.state, "", history, gr.update(value="")
        
        try:            
            system_prompt = self._llm_request_handler(text) if self.use_llm_handler else self._regex_request_handler(text)
            
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
            if last_response:
                # use our tracked language state instead of dropdown value                
                tts_output = self._generate_tts(
                    last_response,
                    language=self.current_tts_lang
                )
                yield history, tts_output, self.home.state, "Respuesta generada con audio.", history, gr.update(value="")
            else:
                yield history, None, self.home.state, "Respuesta generada.", history, gr.update(value="")
            
        except Exception as e:
            error_msg = f"Error al procesar la solicitud: {str(e)}"
            self.logger.error(error_msg)
            
            if history and history[-1]["role"] == "assistant":
                history[-1]["content"] = f"Lo siento, ocurrió un problema: {str(e)}"
            else:
                history.append({"role": "assistant", "content": f"Lo siento, ocurrió un problema: {str(e)}"})
            
            yield history, None, self.home.state, error_msg, history, gr.update(value="")
    
    def process_voice_message(self, audio, history: List):
        """process a voice command and generate response"""
        if audio is None:
            yield history, None, self.home.state, "No se detectó audio.", history, gr.update(value="")
            return
        
        try:
            # transcribe the audio using Whisper ASR
            transcription = self.whisper_asr.transcribe(audio)
            
            if not transcription or transcription.strip() == "":
                yield history, None, self.home.state, "No se ha podido transcribir el audio.", history, gr.update(value="")
                return
            
            history = history or []
            history.append({"role": "user", "content": f"[Voz] {transcription}"})
            yield history, None, self.home.state, "Audio transcrito con éxito.", history, gr.update(value="")
            
            system_prompt = self._llm_request_handler(transcription) if self.use_llm_handler else self._regex_request_handler(transcription)
            
            tts_output = None
            last_response = ""
            
            for partial_response in self.chat_llm.generate_streaming(transcription, system_prompt=system_prompt):
                last_response = partial_response
                
                if history and history[-1]["role"] == "assistant":
                    history[-1]["content"] = partial_response
                else:
                    history.append({"role": "assistant", "content": partial_response})
                
                time.sleep(0.05)  # small delay for visual feedback
                yield history, None, self.home.state, "Procesando respuesta...", history, gr.update(value="")
            
            if last_response:
                # use our tracked language state instead of dropdown value                
                tts_output = self._generate_tts(
                    last_response,
                    language=self.current_tts_lang
                )
                yield history, tts_output, self.home.state, "Respuesta generada con audio.", history, gr.update(value="")
            else:
                yield history, None, self.home.state, "Respuesta generada.", history, gr.update(value="")
            
        except Exception as e:
            error_msg = f"Error al procesar comando de voz: {str(e)}"
            self.logger.error(error_msg)
            
            if history and history[-1]["role"] == "assistant":
                history[-1]["content"] = f"Lo siento, ocurrió un problema: {str(e)}"
            else:
                if not history:
                    history = [
                        {"role": "user", "content": f"[Voz] (Error en transcripción)"},
                    ]
                history.append({"role": "assistant", "content": f"Lo siento, ocurrió un problema: {str(e)}"})
            
            yield history, None, self.home.state, error_msg, history, gr.update(value="")

    def _generate_tts(
        self, 
        text: str, 
        language: str = "spa"
    ) -> Optional[tuple]:
        """generate speech from text with customizable parameters"""
        if not text:
            return None
        
        try:
            result = self.tts_model.generate(
                text=text,
                language=language,
                speaking_rate=1.0,
                noise_scale=0.667,
                noise_scale_duration=0.8,
                save_path=None
            )
            
            # return the audio_out tuple directly (sample_rate, waveform)
            return result["audio_out"]
        except Exception as e:
            self.logger.error(f"Error en generación de voz: {str(e)}")
            return None
     
    def update_system_prompt(self, new_prompt: str, clear_history: bool, current_history: List) -> Tuple[List, str, List]:
        """update the system prompt and optionally clear chat history"""
        self.system_prompt = new_prompt
        
        if clear_history:
            updated_history = []
            updated_chat_state = []
            
            if hasattr(self.chat_llm, "conv_history"):
                self.chat_llm.conv_history = []
                
            log_message = "Prompt del sistema actualizado y chat limpiado."
        else:
            updated_history = current_history
            updated_chat_state = current_history
            log_message = "Prompt del sistema actualizado, chat conservado."
            
        return updated_history, log_message, updated_chat_state

    def clear_chat_history(self) -> Tuple[List, List, str]:
        """clear the chat history"""
        if hasattr(self.chat_llm, "conv_history"):
            self.chat_llm.conv_history = []
        
        return [], [], "Chat borrado con éxito."

    def _update_processing_method(self, use_llm: bool) -> str:
        """update the command processing method"""
        # if the language is not Spanish, force LLM to True regardless of the input
        if self.current_tts_lang != "spa":
            self.use_llm_handler = True
            return "Procesado LLM forzado: solo se puede desactivar en idioma español."
        
        # if the language is Spanish, allow changing the method
        self.use_llm_handler = use_llm
        if use_llm:
            return "Cambio a procesado de peticiones con LLM."
        else:
            return "Cambio a procesado de peticiones con Regex."

    def _switch_llm_model(self, new_model_name: str) -> str:
        """switch to a different LLM model"""
        try:
            if new_model_name not in ChatLLM.models:
                return f"Error: Modelo '{new_model_name}' no disponible. Opciones válidas: {list(ChatLLM.models.keys())}"
            
            # if it's the same model, no need to switch
            if hasattr(self, 'chat_llm') and self.chat_llm.model_name == new_model_name:
                return f"El modelo {new_model_name} ya está en uso."
            
            # store conversation history to preserve it
            old_history = []
            if hasattr(self, 'chat_llm') and self.chat_llm.conv_history:
                old_history = self.chat_llm.conv_history.copy()
            
            # properly unload the current model if it exists
            if hasattr(self, 'chat_llm'):
                self.chat_llm.unload()
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            # initialize and load the new model
            self.chat_llm = ChatLLM(model_name=new_model_name)
            self.chat_llm.load()
            
            # restore conversation history if it existed
            if old_history:
                self.chat_llm.conv_history = old_history
            
            return f"Nuevo modelo actualizado: {new_model_name}"
            
        except Exception as e:
            error_msg = f"Error al actualizar el modelo: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    def _switch_tts_language(self, language: str) -> str:
        """handle TTS language change and update the model if necessary"""
        try:
            # update our internal variable to track the current language
            self.current_tts_lang = language
            self.logger.info(f"Idioma actualizado a: {language}")
            
            # update LLM checkbox state based on language
            # if language is not Spanish, force LLM processing and disable the checkbox
            if language != "spa":
                self.use_llm_handler = True
                # configuración visual del checkbox se actualizará mediante el retorno
                checkbox_update = gr.update(value=True, interactive=False)
                self.logger.info("Procesamiento LLM forzado (idioma no español)")
            else:
                # if language is Spanish, allow user to choose
                checkbox_update = gr.update(interactive=True)
                self.logger.info("Checkbox de procesamiento LLM habilitado (idioma español)")
            
            if not hasattr(self, 'tts_model') or not self.tts_model.is_loaded:
                self.tts_model = VITTS(language=language)
                self.tts_model.load()
                message = f"Modelo TTS inicializado en {language}"
            
            # if the language is the same, no need to switch
            elif self.tts_model.language == language:
                message = f"Modelo TTS ya está configurado en {language}"
            else:
                # update the model with the new language
                self.logger.info(f"Cambiando idioma TTS de {self.tts_model.language} a {language}")
                
                # properly unload the current model to free memory
                self.tts_model.unload()
                
                # clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # create a new model with the updated language
                self.tts_model = VITTS(language=language)
                self.tts_model.load()
                message = f"Idioma TTS cambiado a: {language}"
            
            return message, checkbox_update
            
        except Exception as e:
            error_msg = f"Error al cambiar el idioma TTS: {str(e)}"
            self.logger.error(error_msg)
            return error_msg, gr.update()

    def launch(self, share=True):
        """launch the gradio interface"""
        self.demo.launch(share=share)

# example usage
if __name__ == "__main__":
    assistant = Assistant()
    assistant.launch() 