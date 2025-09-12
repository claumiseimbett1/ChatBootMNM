# swimming_club_bot.py
import streamlit as st
import torch
#from transformers import AutoTokenizer, AutoModelForCausalLM
# Deshabilitar quantización en Windows
QUANTIZATION_AVAILABLE = False
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import pickle
import json
import hashlib
from datetime import datetime
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

@st.cache_resource
def load_llama_model():
    """Carga el modelo con optimizaciones"""
    try:
        # Usar modelo más compatible y liviano
        model_name = "microsoft/DialoGPT-small"  # Más liviano y estable
        
        print(f"📥 Descargando modelo: {model_name}")
        
        # Cargar tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configurar argumentos del modelo - sin device_map para mayor compatibilidad
        model_kwargs = {
            "torch_dtype": torch.float32,  # Cambiar a float32 para mayor compatibilidad
        }
        
        # Cargar modelo
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        print("✅ Modelo cargado exitosamente")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error cargando el modelo: {str(e)}")
        st.error(f"Error cargando el modelo: {str(e)}")
        return None, None

@st.cache_resource
def setup_rag_system(pdf_folder="pdfs"):
    """Configura el sistema RAG con los PDFs del club"""
    try:
        # Embeddings en español
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # Verificar si existe vectorstore guardado
        vectorstore_path = "club_vectorstore"
        if os.path.exists(f"{vectorstore_path}.faiss"):
            vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
            return vectorstore, embeddings
        
        # Si no existe, crear nuevo vectorstore
        all_documents = []
        if os.path.exists(pdf_folder):
            for pdf_file in os.listdir(pdf_folder):
                if pdf_file.endswith('.pdf'):
                    try:
                        pdf_path = os.path.join(pdf_folder, pdf_file)
                        loader = PyPDFLoader(pdf_path)
                        documents = loader.load()
                        
                        for doc in documents:
                            doc.metadata.update({
                                "source": pdf_file,
                                "doc_type": identify_doc_type(pdf_file)
                            })
                        
                        all_documents.extend(documents)
                    except Exception as e:
                        st.warning(f"Error cargando {pdf_file}: {str(e)}")
                        continue
        
        if all_documents:
            # Dividir documentos en chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            texts = text_splitter.split_documents(all_documents)
            vectorstore = FAISS.from_documents(texts, embeddings)
            
            # Guardar para uso futuro
            vectorstore.save_local(vectorstore_path)
        else:
            vectorstore = None
        
        return vectorstore, embeddings
    except Exception as e:
        st.error(f"Error configurando sistema RAG: {str(e)}")
        return None, None

def identify_doc_type(filename):
    """Identifica el tipo de documento"""
    filename_lower = filename.lower()
    if "reglamento" in filename_lower:
        return "reglamento"
    elif "inscripcion" in filename_lower:
        return "inscripcion"
    elif "precios" in filename_lower:
        return "precios"
    else:
        return "general"

class RedisCache:
    """Maneja el cache de respuestas con Redis"""
    def __init__(self):
        self.redis_client = None
        self.cache_available = False
        self._connect()
    
    def _connect(self):
        """Conecta a Redis con fallback"""
        if not REDIS_AVAILABLE:
            print("⚠️ Módulo redis no instalado - cache deshabilitado")
            self.cache_available = False
            return
            
        try:
            # Intentar conexión local primero
            self.redis_client = redis.Redis(
                host='localhost', 
                port=6379, 
                db=0, 
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2
            )
            # Probar conexión
            self.redis_client.ping()
            self.cache_available = True
            print("✅ Redis conectado exitosamente")
        except Exception as e:
            print(f"⚠️ Redis no disponible: {e}")
            self.cache_available = False
    
    def _generate_key(self, user_input):
        """Genera clave única para el cache"""
        normalized_input = user_input.lower().strip()
        return f"chatbot_response:{hashlib.md5(normalized_input.encode()).hexdigest()}"
    
    def get_response(self, user_input):
        """Obtiene respuesta del cache"""
        if not self.cache_available:
            return None
        
        try:
            key = self._generate_key(user_input)
            cached_data = self.redis_client.get(key)
            if cached_data:
                response_data = json.loads(cached_data)
                print(f"🔄 Cache HIT para: {user_input[:50]}...")
                return response_data['response']
        except Exception as e:
            print(f"Error obteniendo del cache: {e}")
        
        return None
    
    def set_response(self, user_input, response, ttl=3600):
        """Guarda respuesta en cache (TTL: 1 hora por defecto)"""
        if not self.cache_available:
            return False
        
        try:
            key = self._generate_key(user_input)
            response_data = {
                'response': response,
                'timestamp': datetime.now().isoformat(),
                'input': user_input
            }
            self.redis_client.setex(key, ttl, json.dumps(response_data))
            print(f"💾 Cache SAVE para: {user_input[:50]}...")
            return True
        except Exception as e:
            print(f"Error guardando en cache: {e}")
            return False
    
    def clear_cache(self):
        """Limpia todo el cache del chatbot"""
        if not self.cache_available:
            return False
        
        try:
            keys = self.redis_client.keys("chatbot_response:*")
            if keys:
                self.redis_client.delete(*keys)
                print(f"🗑️ Cache limpiado: {len(keys)} entradas eliminadas")
            return True
        except Exception as e:
            print(f"Error limpiando cache: {e}")
            return False

class LlamaSwimmingBot:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.vectorstore = None
        self.embeddings = None
        self.conversation_history = []
        self.cache = RedisCache()
    
    def search_documents(self, query, k=2):
        """Busca en los documentos PDF"""
        try:
            if not self.vectorstore:
                print("⚠️ No hay vectorstore disponible")
                return ""
            
            print(f"🔍 Buscando: {query}")
            docs = self.vectorstore.similarity_search(query, k=k)
            context = ""
            for doc in docs:
                context += f"\n[{doc.metadata.get('doc_type', 'documento')}]: {doc.page_content}\n"
            
            print(f"📄 Documentos encontrados: {len(docs)}")
            return context
        except Exception as e:
            print(f"❌ Error en búsqueda de documentos: {e}")
            return ""
    
    def get_enrollment_flow(self, step=1):
        """Maneja el flujo de inscripción paso a paso"""
        enrollment_steps = {
            1: """✅ **PERFECTO, ESTOS SON LOS PASOS PARA INSCRIBIRTE:**

1️⃣ **Actividad:** Ofrecemos clases de natación en la piscina olímpica de la Villa Olímpica de Montería.
Es un deporte de bajo impacto, ideal para la salud.

2️⃣ **Requisitos:**
• Aceptar términos y condiciones
• Firmar consentimiento informado
• Presentar certificado médico, si es necesario

3️⃣ **Matrícula:**
• Tiene vigencia de 1 año
• Solo se paga una vez al año
• Importante: No se devuelve el valor pagado

4️⃣ **Mensualidad:**
• Se paga por adelantado cada mes
• Solo puedes asistir si estás al día en el pago
• Tarifa pronto pago los primeros 5 dias del ciclo


📆 **¿Cómo es la política de devoluciones?**
🟡 Antes de la primera clase: 100%
🟠 Después de la segunda clase: 50%
🔴 Después de la tercera clase: No hay devolución

5️⃣ **Importante sobre el uso de la piscina**
La piscina es pública. El aporte mensual garantiza instructores calificados, no es el alquiler del espacio.

6️⃣ **¿Qué riesgos debo tener en cuenta?**
• Lesiones menores, ahogamiento, contacto con otros usuarios, clima
• Declaras estar en condiciones óptimas de salud
• Si representas a un menor, también asumes responsabilidad por él/ella

7️⃣ **¿Se toman fotos o videos?**
Sí. Autorizas su uso con fines deportivos y promocionales del club al aceptar los términos.

8️⃣ **¿Deseas continuar con tu inscripción?**

✅ **Sí, quiero inscribirme**
❌ **No, volver al inicio**
📩 **Contactar asesor humano para solicitar la documentación**

9️⃣ **¿Tienes dudas sobre nuestra política de reposición de clases?**
📋 Pregúntame específicamente sobre "política de reposición" o "reponer clases" para obtener información detallada.

📞 **WhatsApp:** +57 3144809367
📧 **Email:** monteriamaster@gmail.com
👆 [Haz clic aquí para inscribirte por WhatsApp](https://wa.me/573144809367?text=Hola,%20quiero%20inscribirme%20en%20el%20Club%20de%20Natación%20MNM)
💌 [Enviar correo electrónico](mailto:monteriamaster@gmail.com?subject=Inscripción%20Club%20de%20Natación%20MNM)"""
        }
        
        return enrollment_steps.get(step, "Paso no válido")

    def get_fallback_response(self, user_input):
        """Respuesta de emergencia usando la información hardcodeada"""
        user_lower = user_input.lower()
        
        # Detectar solicitudes de inscripción
        if any(word in user_lower for word in ["inscripcion", "inscripción", "inscribirme", "matricula", "matrícula", "registro", "pasos", "como me inscribo", "cómo me inscribo"]):
            return self.get_enrollment_flow(1)
        
        if any(word in user_lower for word in ["horario", "hora", "cuando", "tiempo"]):
            if any(word in user_lower for word in ["niño", "niña", "menor", "infantil"]):
                return """🏊‍♀️ **HORARIOS PARA NIÑOS:**

**Martes y Jueves:**
• 4:00 PM a 5:00 PM
• 5:00 PM a 6:00 PM

**Sábados:**
• 8:00 AM a 9:00 AM
• 4:00 PM a 5:00 PM
• 5:00 PM a 6:00 PM

**Miércoles y Viernes:**
• 4:00 PM a 5:00 PM
• 5:00 PM a 6:00 PM

📞 **WhatsApp:** +57 3144809367
📧 **Email:** monteriamaster@gmail.com

🔥 **¡REALIZA TU INSCRIPCIÓN YA!**
👆 [Haz clic aquí para inscribirte por WhatsApp](https://wa.me/573144809367?text=Hola,%20quiero%20inscribirme%20en%20el%20Club%20de%20Natación%20MNM)
💌 [Enviar correo electrónico](mailto:monteriamaster@gmail.com?subject=Inscripción%20Club%20de%20Natación%20MNM)"""
            
            elif any(word in user_lower for word in ["adulto", "mayor"]):
                return """🏊‍♂️ **HORARIOS PARA ADULTOS:**

**Martes y Jueves:**
• 5:00 AM a 6:00 AM
• 6:00 AM a 7:00 AM
• 7:00 AM a 8:00 AM
• 6:00 PM a 7:00 PM
• 7:00 PM a 8:00 PM

**Sábados:**
• 5:00 AM a 6:00 AM
• 6:00 AM a 7:00 AM
• 7:00 AM a 8:00 AM

**Miércoles y Viernes:**
• 6:00 PM a 7:00 PM

📞 **WhatsApp:** +57 3144809367
📧 **Email:** monteriamaster@gmail.com

🔥 **¡REALIZA TU INSCRIPCIÓN YA!**
👆 [Haz clic aquí para inscribirte por WhatsApp](https://wa.me/573144809367?text=Hola,%20quiero%20inscribirme%20en%20el%20Club%20de%20Natación%20MNM)
💌 [Enviar correo electrónico](mailto:monteriamaster@gmail.com?subject=Inscripción%20Club%20de%20Natación%20MNM)"""
            
            else:
                return """📅 **HORARIOS COMPLETOS - CLUB DE NATACIÓN MNM:**

**MARTES Y JUEVES:**
• 5:00-8:00 AM (adultos)
• 4:00-6:00 PM (niños)
• 6:00-8:00 PM (adultos)

**SÁBADOS:**
• 5:00-8:00 AM (adultos)
• 8:00 AM-6:00 PM (niños y adultos)

**MIÉRCOLES Y VIERNES:**
• 4:00-6:00 PM (niños)
• 6:00-7:00 PM (adultos)

📍 Ubicación: Piscina de la Villaolímpica, Montería
📞 WhatsApp: +57 3144809367

🔥 **¡REALIZA TU INSCRIPCIÓN YA!**
👆 [Haz clic aquí para inscribirte por WhatsApp](https://wa.me/573144809367?text=Hola,%20quiero%20inscribirme%20en%20el%20Club%20de%20Natación%20MNM)"""
        
        elif any(word in user_lower for word in ["precio", "costo", "valor", "cuanto", "pago"]):
            return """💰 **PRECIOS CLUB DE NATACIÓN MNM:**

🏊‍♀️ **MENSUALIDADES:**
1. 1️⃣  vez por semana: $120,000
2. 2️⃣  veces por semana: $160,000
3. 3️⃣  veces por semana: $180,000

💡 **Tarifa con descuento pronto pago:** Los primeros 5 días del ciclo

📝 **Inscripción:** $40,000 (pago único)

📞 **WhatsApp:** +57 3144809367
📧 **Email:** monteriamaster@gmail.com

🔥 **¡REALIZA TU INSCRIPCIÓN YA!**
👆 [Haz clic aquí para inscribirte por WhatsApp](https://wa.me/573144809367?text=Hola,%20quiero%20inscribirme%20en%20el%20Club%20de%20Natación%20MNM)
💌 [Enviar correo electrónico](mailto:monteriamaster@gmail.com?subject=Inscripción%20Club%20de%20Natación%20MNM)"""
        
        elif any(word in user_lower for word in ["traer", "necesito", "llevar", "primera clase", "equipamiento"]):
            return """🎒 **QUÉ TRAER A TU PRIMERA CLASE:**

✅ **Obligatorio:**
• Traje de baño deportivo
• Gorro de natación
• Gafas de natación
• Toalla

✅ **Opcional:**
• Chanclas antideslizantes

👶 **Edades:** Desde 5 años sin límite superior

📞 **WhatsApp:** +57 3144809367
📧 **Email:** monteriamaster@gmail.com

🔥 **¡REALIZA TU INSCRIPCIÓN YA!**
👆 [Haz clic aquí para inscribirte por WhatsApp](https://wa.me/573144809367?text=Hola,%20quiero%20inscribirme%20en%20el%20Club%20de%20Natación%20MNM)
💌 [Enviar correo electrónico](mailto:monteriamaster@gmail.com?subject=Inscripción%20Club%20de%20Natación%20MNM)"""
        
        elif any(word in user_lower for word in ["enfasis", "énfasis", "enfoque", "que enseñan", "metodologia", "metodología", "escuela", "enseñanza", "sistema", "niveles", "como enseñan", "que aprendo", "qué aprendo"]):
            return """🎯 **ÉNFASIS DE NUESTRA ESCUELA:**

1. 🏊‍♀️ Desarrollo de habilidades acuáticas
2. 🏊‍♂️ Enseñanza de técnicas en los 4 estilos de natación
3. 📊 Sistema de evaluación progresivo por niveles (nivel basico, intermedio, avanzado y equipo)
4. 🏆 Programa de reconocimiento del Nadador del trimestre
5. 📈 Evaluación mensual del avance del nivel con puntaje que es enviado al grupo de Practicantes del Club
6. 💪 Entrenamiento para resistencia y velocidad
7. 👥 Natación para todas las edades
8. 🥇 Preparación para competencias
9. ⚡ Fomento de disciplina y trabajo en equipo
10. 🌱 Promoción de estilo de vida saludable

📞 **WhatsApp:** +57 3144809367
📧 **Email:** monteriamaster@gmail.com

🔥 **¡REALIZA TU INSCRIPCIÓN YA!**
👆 [Haz clic aquí para inscribirte por WhatsApp](https://wa.me/573144809367?text=Hola,%20quiero%20inscribirme%20en%20el%20Club%20de%20Natación%20MNM)
💌 [Enviar correo electrónico](mailto:monteriamaster@gmail.com?subject=Inscripción%20Club%20de%20Natación%20MNM)"""
        
        elif any(word in user_lower for word in ["edad", "años", "niño", "menor"]):
            return """<div style="text-align: center;">

👶 **EDADES ACEPTADAS:**

✅ Desde 5 años sin límite superior

🏊‍♀️ Tenemos horarios especializados para niños y adultos, en grupos segmentados para facilitar y promover el aprendizaje

</div>

📞 **WhatsApp:** +57 3144809367
📧 **Email:** monteriamaster@gmail.com

🔥 **¡REALIZA TU INSCRIPCIÓN YA!**
👆 [Haz clic aquí para inscribirte por WhatsApp](https://wa.me/573144809367?text=Hola,%20quiero%20inscribirme%20en%20el%20Club%20de%20Natación%20MNM)
💌 [Enviar correo electrónico](mailto:monteriamaster@gmail.com?subject=Inscripción%20Club%20de%20Natación%20MNM)"""
        
        elif any(word in user_lower for word in ["contacto", "teléfono", "telefono", "whatsapp", "direccion", "dirección", "ubicacion", "ubicación", "donde"]):
            return """📍 **INFORMACIÓN DE CONTACTO:**

🏊‍♀️ **Club de Natación Montería Natación Master**
📍 Dirección: Piscina de la Villaolímpica, Montería
📞 Teléfono: +57 3144809367
💬 WhatsApp: +57 3144809367
📧 Email: monteriamaster@gmail.com

¡Te esperamos! 🌊"""
        
        elif any(word in user_lower for word in ["reposicion", "reposición", "reponer", "recuperar clase", "faltar"]):
            return """📋 **POLÍTICA DE REPOSICIÓN - MONTERÍA NATACIÓN MASTER**

✅ Entendemos que a veces surgen imprevistos. Por eso, puedes reponer una (1) clase por mes, y evaluamos cada caso según la justificación que nos compartas.

📅 Puedes tomar tu reposición en otro horario dentro del mismo mes, en grupos del mismo nivel y calendario, según disponibilidad de cupo. Si faltaste en la última semana del ciclo, ¡tranqui! tienes hasta 8 días del mes siguiente para recuperarla.

🔁 Ten en cuenta que las reposiciones no se acumulan ni se trasladan a otros meses.

🌧 Si la piscina se cierra por motivos externos, garantizamos las reposiciones que correspondan.

❌ Para cuidar la organización de nuestros grupos y ofrecerte una buena experiencia, no reponemos clases sin aviso previo.

📲 Escríbenos por WhatsApp oficial del club para gestionar tu reposición. ¡Estamos para ayudarte! 🏊‍♀️✨

📎 **Documento completo:** https://bit.ly/32J20r0

📞 **WhatsApp:** +57 3144809367
📧 **Email:** monteriamaster@gmail.com

🔥 **¡REALIZA TU INSCRIPCIÓN YA!**
👆 [Haz clic aquí para inscribirte por WhatsApp](https://wa.me/573144809367?text=Hola,%20quiero%20inscribirme%20en%20el%20Club%20de%20Natación%20MNM)
💌 [Enviar correo electrónico](mailto:monteriamaster@gmail.com?subject=Inscripción%20Club%20de%20Natación%20MNM)"""
        
        elif any(word in user_lower for word in ["reglamento", "reglas", "normas", "politicas", "políticas", "terminos", "términos", "condiciones"]):
            return """📋 **INFORMACIÓN SOBRE REGLAMENTOS:**

Para información detallada sobre:
• Reglamentos del club
• Políticas de reposición
• Términos y condiciones
• Normas de convivencia

📞 Por favor contacta directamente al club:
WhatsApp: +57 3144809367

Tenemos documentación completa disponible."""
        
        elif any(word in user_lower for word in ["inscripcion", "inscripción", "matricula", "matrícula", "registro"]):
            return """📝 **PROCESO DE INSCRIPCIÓN:**

💰 **Costo de inscripción:** $40,000 (pago único)

📋 Para completar tu inscripción necesitas:
• Documentación personal
• Información médica básica
• Selección de horarios

📞 Para iniciar el proceso contactanos:
WhatsApp: +57 3144809367

¡Te ayudaremos con todo el proceso, Bienvenido! 🏊‍♀️"""
        
        return None

    def generate_response(self, user_input):
        """Genera respuesta usando fallback primero, luego PDFs si es necesario"""
        # DEBUG: Log de entrada
        print(f"🔍 INPUT: {user_input}")
        
        # Intentar obtener respuesta del cache primero
        cached_response = self.cache.get_response(user_input)
        if cached_response:
            print("📋 Usando respuesta del cache")
            return cached_response
        
        # Primero intentar respuesta de fallback (respuestas hardcodeadas)
        fallback = self.get_fallback_response(user_input)
        if fallback:
            print("✅ Respuesta hardcodeada encontrada")
            # Guardar en cache con TTL largo para respuestas frecuentes
            self.cache.set_response(user_input, fallback, ttl=7200)  # 2 horas
            return fallback
        
        # Si no hay respuesta hardcodeada, buscar en PDFs
        print("📚 No hay respuesta hardcodeada, buscando en PDFs...")
        document_context = self.search_documents(user_input)
        print(f"📄 Contexto encontrado: {len(document_context) if document_context else 0} caracteres")
        
        if document_context and len(document_context.strip()) > 50:
            # Si encontramos contenido relevante en PDFs, devolver eso
            pdf_response = f"""📋 **Información encontrada en documentos del club:**

{document_context}

📞 **WhatsApp:** +57 3144809367
📧 **Email:** monteriamaster@gmail.com
🔥 **¡REALIZA TU INSCRIPCIÓN YA!**
👆 [Haz clic aquí para inscribirte por WhatsApp](https://wa.me/573144809367?text=Hola,%20quiero%20inscribirme%20en%20el%20Club%20de%20Natación%20MNM)
💌 [Enviar correo electrónico](mailto:monteriamaster@gmail.com?subject=Consulta%20Club%20de%20Natación%20MNM)"""
            
            print("✅ Respuesta generada desde PDFs")
            self.cache.set_response(user_input, pdf_response, ttl=3600)  # 1 hora
            return pdf_response
        
        # Si no encontramos nada en PDFs, dar respuesta genérica
        print("❌ No se encontró información específica")
        generic_response = f"""🏊‍♀️ **Club de Natación Montería Natación Master**

Lo siento, no tengo información específica sobre tu consulta en este momento.

📞 Para información detallada contacta directamente:
💬 WhatsApp: +57 3144809367
📧 Email: monteriamaster@gmail.com
📍 Piscina de la Villaolímpica, Montería

🔥 **¡REALIZA TU INSCRIPCIÓN YA!**
👆 [Haz clic aquí para inscribirte por WhatsApp](https://wa.me/573144809367?text=Hola,%20quiero%20inscribirme%20en%20el%20Club%20de%20Natación%20MNM)
💌 [Enviar correo electrónico](mailto:monteriamaster@gmail.com?subject=Consulta%20Club%20de%20Natación%20MNM)

¡Estaremos felices de ayudarte! 🌊"""
        
        self.cache.set_response(user_input, generic_response, ttl=1800)  # 30 minutos
        return generic_response

    def old_llama_method(self, user_input, document_context):
        """Método anterior con Llama (mantenido por si se necesita)"""
        # Información base del club
        club_info = """
CLUB DE NATACION MONTERIA NATACIÓN MASTER
- Dirección: Piscina de la Villaolimpica, Monteria
- Teléfono: +57 3144809367
- WhatsApp: +57 3144809367
- Edades: Desde 5 años sin límite superior
- Horarios: Martes, Jueves: 5:00 AM a 6:00 AM (horario de adultos)
6:00 AM a 7:00 AM (horario de adultos)
7:00 AM a 8:00 AM (horario de adultos)
4:00 PM a 5:00 PM (horario de niños)
5:00 PM a 6:00 PM (horario de niños)
6:00 PM a 7:00 PM (horario de adultos)
7:00 PM a 8:00 PM (horario de adultos)
Sábado: 5:00 AM a 6:00 AM (horario de adultos)
6:00 AM a 7:00 AM (horario de adultos)
7:00 AM a 8:00 AM (horario de adultos)
8:00 AM a 9:00 AM (horario de niños)
4:00 PM a 5:00 PM (horario de niños)
5:00 PM a 6:00 PM (horario de niños)
Miércoles, Viernes: 4:00 PM a 5:00 PM (horario de niños)
5:00 PM a 6:00 PM (horario de niños)
6:00 PM a 7:00 PM (horario de adultos)

CUÁL ES EL ENFASIS DE LA ESCUELA:
- Desarrollo de habilidades acuáticas
- Enseñanza de técnicas de natación en los cuatro estilos
- Sistema de evaluación progresivo por niveles (inicial, intermedio, avanzado y equipo)
- Entrenamiento para mejorar resistencia y velocidad
- Enseñanza de natación para todas las edades
- Preparación para competencias y eventos
- Fomento de la disciplina y el trabajo en equipo
- Promoción de un estilo de vida saludable

PRECIOS:
- $120,000 mensuales, una vez por semana (pronto pago los primeros 5 dias del ciclo)
- $160,000 mensuales, dos veces por semana (pronto pago los primeros 5 dias del ciclo)
- $180,000 mensuales, tres veces por semana (pronto pago los primeros 5 dias del ciclo)
- Inscripción: $40,000 (única vez)

QUÉ TRAER PRIMERA CLASE:
- Traje de baño deportivo, gorro, gafas de natación
- Toalla y opcional chanclas antideslizantes
        """
        
        # Historial de conversación
        conversation_context = "\n".join(self.conversation_history[-4:])
        
        # Crear prompt compatible con DialoGPT
        prompt = f"""Club de Natación Monteria - Información disponible:

{club_info}

{document_context}

Conversación:
Usuario: {user_input}
Asistente:"""

        # Generar respuesta
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                early_stopping=True,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decodificar respuesta
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response.split("Asistente:")[-1].strip()
        
        # Filtrar respuestas problemáticas y usar fallback si es necesario
        problematic_phrases = [
            "contacta al club", "contacta al personal", "comunícate con", 
            "llama al club", "no tengo información", "no puedo proporcionar",
            "consulta directamente", "ponte en contacto"
        ]
        
        if any(phrase in response.lower() for phrase in problematic_phrases):
            fallback_response = self.get_fallback_response(user_input)
            if fallback_response:
                response = fallback_response
            else:
                response = f"""🏊‍♀️ **Club de Natación Montería Natación Master**

Para información específica sobre tu consulta:
📞 WhatsApp: +57 3144809367
📍 Piscina de la Villaolímpica, Montería

🔥 **¡REALIZA TU INSCRIPCIÓN YA!**
👆 [Haz clic aquí para inscribirte por WhatsApp](https://wa.me/573144809367?text=Hola,%20quiero%20inscribirme%20en%20el%20Club%20de%20Natación%20MNM)

¡Estaremos felices de ayudarte! 🌊"""
        
        # Guardar respuesta en cache con TTL menor para respuestas del modelo
        self.cache.set_response(user_input, response, ttl=1800)  # 30 minutos
        
        # Actualizar historial
        self.conversation_history.append(f"Usuario: {user_input}")
        self.conversation_history.append(f"Asistente: {response}")
        
        return response

# Aplicación Streamlit
def main():
    st.set_page_config(
        page_title="CHATBOOTMNM",
        page_icon="🏊‍♀️",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Meta viewport para móvil
    st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    """, unsafe_allow_html=True)
    
    # CSS personalizado completo con optimización móvil
    st.markdown("""
    <style>
    /* Configuración general */
    .stApp {
        background-color: #1e3a5f;
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Forzar texto blanco en toda la aplicación */
    .stApp * {
        color: #ffffff !important;
    }
    
    /* Header con logo */
    .header-container {
        background: linear-gradient(135deg, #1a3d70 0%, #134492 100%);
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(26, 61, 112, 0.2);
        text-align: center;
    }
    
    .logo-space {
        width: 60px;
        height: 60px;
        background-color: #ffffff;
        border-radius: 50%;
        margin: 0 auto 10px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 30px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
    }
    
    .main-title {
        color: #ffffff !important;
        font-size: 22px;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        line-height: 1.3;
    }
    
    /* Contenedor principal */
    .main-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 10px;
    }
    
    /* Botones de consultas rápidas */
    .stButton > button {
        background: linear-gradient(135deg, #134492 0%, #1a3d70 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 14px 16px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 3px 8px rgba(19, 68, 146, 0.3);
        width: 100%;
        margin-bottom: 8px;
        font-size: 14px;
        text-align: center;
        min-height: 48px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #dede3c 0%, #1a3d70 100%);
        transform: translateY(-1px);
        box-shadow: 0 5px 15px rgba(222, 222, 60, 0.4);
    }
    
    /* Mensajes de chat */
    .chat-message {
        padding: 12px;
        border-radius: 15px;
        margin-bottom: 12px;
        display: flex;
        align-items: flex-start;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background: linear-gradient(135deg, #dede3c 0%, #f5f5a3 100%);
        margin-left: 10%;
        color: #1a3d70 !important;
        font-weight: 600;
        text-shadow: none;
        box-shadow: 0 3px 10px rgba(222, 222, 60, 0.3);
        border: 2px solid #134492;
    }
    
    .user-message * {
        color: #1a3d70 !important;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #134492 0%, #1a3d70 100%);
        margin-right: 10%;
        color: #ffffff !important;
        box-shadow: 0 3px 10px rgba(19, 68, 146, 0.3);
        border: 2px solid #dede3c;
    }
    
    .bot-message * {
        color: #ffffff !important;
    }
    
    /* Input de texto */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #134492;
        padding: 14px 20px;
        background-color: #ffffff;
        color: #1a3d70 !important;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #1a3d70;
        box-shadow: 0 0 15px rgba(26, 61, 112, 0.3);
        outline: none;
    }
    
    /* Spinner personalizado */
    .stSpinner > div {
        border-top-color: #134492 !important;
    }
    
    /* Sección de consultas frecuentes */
    .frequent-queries {
        background: linear-gradient(rgba(255, 255, 255, 0.95), rgba(255, 255, 255, 0.95));
        padding: 15px;
        border-radius: 15px;
        margin: 15px 0;
        border: 3px solid #134492;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        max-width: 100%;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Título de secciones */
    .section-title {
        color: #1a3d70 !important;
        background-color: rgba(255, 255, 255, 0.95);
        font-size: 20px;
        font-weight: 900;
        margin-bottom: 15px;
        text-align: center;
        padding: 8px 12px;
        border-radius: 8px;
        display: inline-block;
        width: 100%;
        box-sizing: border-box;
    }
    
    /* Chat input del chatbot */
    .stChatInput > div {
        background-color: transparent;
    }
    
    .stChatInput input {
        background-color: #ffffff !important;
        color: #1a3d70 !important;
        border: 2px solid #134492 !important;
        border-radius: 25px !important;
        padding: 14px 20px !important;
        font-size: 16px !important;
    }
    
    /* Scrollbar personalizada */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #134492, #1a3d70);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #134492, #1a3d70);
    }
    
    /* Footer o información adicional */
    .info-footer {
        background: linear-gradient(135deg, #1a3d70 0%, #134492 100%);
        color: white;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        margin-top: 30px;
        box-shadow: 0 3px 10px rgba(26, 61, 112, 0.3);
    }
    
    /* MEDIA QUERIES PARA MÓVIL */
    @media screen and (max-width: 768px) {
        .stApp {
            padding: 0 !important;
        }
        
        .header-container {
            padding: 10px;
            margin-bottom: 15px;
            border-radius: 10px;
        }
        
        .main-title {
            font-size: 18px;
            line-height: 1.2;
        }
        
        .logo-space {
            width: 50px;
            height: 50px;
            font-size: 24px;
        }
        
        .section-title {
            font-size: 18px;
            padding: 6px 8px;
        }
        
        .frequent-queries {
            padding: 10px;
            margin: 10px 5px;
            background: linear-gradient(rgba(222, 222, 60, 0.95), rgba(245, 245, 163, 0.95)) !important;
            border: 3px solid #134492 !important;
        }
        
        .stButton > button {
            padding: 16px 12px;
            font-size: 13px;
            min-height: 50px;
            margin-bottom: 6px;
            font-weight: 700 !important;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3) !important;
        }
        
        .chat-message {
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 12px;
        }
        
        .user-message {
            margin-left: 5%;
            font-size: 14px;
        }
        
        .bot-message {
            margin-right: 5%;
            font-size: 14px;
        }
        
        .stTextInput > div > div > input,
        .stChatInput input {
            padding: 12px 16px !important;
            font-size: 16px !important;
            border-radius: 25px !important;
            border: 2px solid #134492 !important;
            background-color: #ffffff !important;
            color: #1a3d70 !important;
            width: 100% !important;
            box-sizing: border-box !important;
            min-height: 48px !important;
        }
        
        /* Columnas en móvil - hacer que los botones ocupen toda la fila */
        .element-container .row-widget.stButton {
            width: 100% !important;
        }
        
        /* Ajustar el spacing entre elementos */
        .element-container {
            margin-bottom: 0.5rem !important;
        }
        
        /* Chat input simplificado para móvil */
        .stChatInput {
            background: #1e3a5f !important;
            padding: 10px !important;
            margin: 10px 0 !important;
            border-radius: 15px !important;
        }
        
        .stChatInput > div {
            width: 100% !important;
        }
        
        /* Mejorar la legibilidad de los mensajes en móvil */
        .chat-message {
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
            hyphens: auto !important;
        }
        
        .chat-message h1, .chat-message h2, .chat-message h3 {
            font-size: clamp(16px, 4vw, 20px) !important;
            margin: 10px 0 8px 0 !important;
        }
        
        .chat-message p {
            font-size: clamp(14px, 3.5vw, 16px) !important;
            line-height: 1.5 !important;
            margin: 8px 0 !important;
        }
        
        .chat-message ul, .chat-message ol {
            padding-left: 20px !important;
            margin: 8px 0 !important;
        }
        
        .chat-message li {
            font-size: clamp(14px, 3.5vw, 16px) !important;
            line-height: 1.4 !important;
            margin: 4px 0 !important;
        }
        
        /* Hacer hover menos agresivo en móvil */
        .element-container:hover {
            transform: none;
        }
        
        .stButton > button:hover {
            transform: none;
        }
    }
    
    /* MEDIA QUERIES PARA PANTALLAS MUY PEQUEÑAS */
    @media screen and (max-width: 480px) {
        .main-title {
            font-size: 16px;
        }
        
        .section-title {
            font-size: 16px;
        }
        
        .stButton > button {
            font-size: 12px;
            padding: 14px 8px;
            min-height: 48px;
        }
        
        .chat-message {
            padding: 8px;
            font-size: 13px;
        }
        
        .user-message,
        .bot-message {
            margin-left: 2%;
            margin-right: 2%;
        }
        
        .frequent-queries {
            margin: 10px 2px;
            padding: 8px;
        }
        
        /* Asegurar funcionalidad del chat input */
        .stChatInput {
            position: relative !important;
            z-index: 1 !important;
        }
        
        .stChatInput input {
            -webkit-user-select: text !important;
            user-select: text !important;
            -webkit-touch-callout: default !important;
            touch-action: manipulation !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header personalizado con logo optimizado para móvil
    col_logo, col_title = st.columns([1, 4])
    
    with col_logo:
        try:
            st.image("logo/LOGO ORIGINAL.png", width=120)
        except:
            st.markdown("""
            <div class="logo-space">🥽</div>
            """, unsafe_allow_html=True)
    
    with col_title:
        st.markdown("""
        <div class="header-container">
            <h1 class="main-title">NatalIA - Asistente Virtual del Club Montería Natación Master (MNM)</h1>
        </div>
        """, unsafe_allow_html=True)
    
    # Mensaje de bienvenida optimizado para móvil
    st.markdown("""
    <div style="background: linear-gradient(135deg, #dede3c 0%, #f5f5a3 100%); 
                padding: 15px; 
                border-radius: 15px; 
                margin: 15px 5px; 
                border: 3px solid #134492;
                box-shadow: 0 5px 15px rgba(19, 68, 146, 0.3);
                text-align: center;">
        <h3 style="color: #1a3d70; margin-bottom: 12px; font-weight: bold; font-size: clamp(18px, 4vw, 24px); text-shadow: 1px 1px 2px rgba(255,255,255,0.7);">¡Hola! Bienvenido al Club Montería Natación Master</h3>
        <p style="color: #134492; font-size: clamp(14px, 3.5vw, 16px); margin: 0; line-height: 1.4; font-weight: 600;">
            Soy tu asistente virtual <strong style="color: #1a3d70;">NatalIA</strong> y te enseñaré todo sobre nuestro club y el proceso de inscripción. Estamos en Villaolimpica - Monteria.
            <br><strong style="color: #1a3d70;">¿Listo para sumergirte en tu proceso de aprendizaje o entrenamiento?</strong> 🏊‍♀️
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Inicializar bot
    if "bot" not in st.session_state:
        with st.spinner("Inicializando sistema..."):
            st.session_state.bot = LlamaSwimmingBot()
            st.session_state.bot.vectorstore, st.session_state.bot.embeddings = setup_rag_system()
        st.success("✅ Sistema listo!")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Botones de consultas rápidas
    st.markdown("""
    <div class="frequent-queries">
        <div class="section-title" style="color: #1a3d70 !important; background: linear-gradient(135deg, #dede3c 0%, #f5f5a3 100%); padding: 12px; border-radius: 10px; font-weight: bold; font-size: clamp(18px, 4vw, 24px); border: 2px solid #134492; box-shadow: 0 3px 8px rgba(19, 68, 146, 0.2);">Consultas Frecuentes</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Layout adaptativo para móvil
    if st.session_state.get("mobile_layout", True):
        # Layout móvil: una sola columna
        col1 = st.container()
        col2 = st.container()
        col3 = st.container()
    else:
        # Layout desktop: tres columnas
        col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📅 Horarios del club"):
            user_input = "¿Cuáles son los horarios de funcionamiento?"
            process_message(user_input)
            
        if st.button("📅 Horarios del club para niños"):
            user_input = "¿Cuáles son los horarios de niños?"
            process_message(user_input) 
            
        if st.button("📅 Horarios del club para adultos"):
            user_input = "¿Cuáles son los horarios de adultos?"
            process_message(user_input)
    
    with col2:
        if st.button("💰 Precios por frecuencia semanal"):
            user_input = "¿Cuáles son los precios del mes por frecuencia semanal?"
            process_message(user_input)
            
        if st.button("🎯 Énfasis de nuestra Escuela de Natación"):
            user_input = "¿Cuál es el énfasis de la Escuela de Natación MNM?"
            process_message(user_input)
            
        if st.button("📝 Pasos para inscripción"):
            user_input = "¿Cómo me inscribo?"
            process_message(user_input)
        
    with col3:
        if st.button("🏊‍♂️ ¿Qué traer en la primera clase?"):
            user_input = "¿Qué debo traer a mi primera clase de natación?"
            process_message(user_input)
        
        if st.button("📋 Política de reposición de clases"):
            user_input = "política de reposición"
            process_message(user_input)
            
        if st.button("👶 Edades aceptadas"):
            user_input = "¿Desde qué edad aceptan niños?"
            process_message(user_input)
    
    # Mostrar historial de chat optimizado para móvil
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message chat-message">
                    <div style="font-size: clamp(14px, 3.5vw, 16px); line-height: 1.4;">
                        {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-message chat-message">
                    <div style="font-size: clamp(14px, 3.5vw, 16px); line-height: 1.4;">
                        {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Input del usuario
    if prompt := st.chat_input("Escribe tu pregunta aquí..."):
        process_message(prompt)

def process_message(user_input):
    """Procesa un mensaje del usuario"""
    # Agregar mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generar respuesta
    with st.spinner("Pensando..."):
        response = st.session_state.bot.generate_response(user_input)
    
    # Agregar respuesta al historial
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Rerun para mostrar los nuevos mensajes
    st.rerun()

if __name__ == "__main__":
    main()