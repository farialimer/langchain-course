from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_vertexai import ChatVertexAI
# from langchain_community.chat_models import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-learning!")
    information = """Jair Messias Bolsonaro GOMM (Glicério,[nota 3] 21 de março de 1955) é um militar reformado e político brasileiro, filiado ao Partido Liberal (PL). Foi o 38.º presidente do Brasil, de 1.º de janeiro de 2019 a 1.º de janeiro de 2023, tendo sido eleito pelo Partido Social Liberal (PSL) nas eleições de 2018. Anteriormente, foi deputado federal pelo Rio de Janeiro entre 1991 e 2018, eleito por diversos partidos. Nasceu em Glicério, mas passou a adolescência em Eldorado, no interior de São Paulo. Começou sua carreira militar no município fluminense de Resende após formar-se na Academia Militar das Agulhas Negras (AMAN) em 1977. Serviu nos grupos de artilharia de campanha e paraquedismo do Exército Brasileiro.

    Tornou-se conhecido do público em 1986, quando escreveu um artigo para a revista Veja criticando os baixos salários dos militares, texto pelo qual foi preso.[10] Em 1987, a mesma revista o acusou de planejar plantar bombas em unidades militares,[10] crime militar pelo qual foi condenado em primeira instância, porém o Superior Tribunal Militar o absolveu no ano seguinte.[11] Transferiu-se para a reserva em 1988 com o posto de capitão[12] e concorreu à Câmara Municipal do Rio de Janeiro, sendo eleito vereador pelo Partido Democrata Cristão (PDC).[13] Em 1990, foi eleito deputado federal,[14] cargo para o qual foi reeleito seis vezes. Durante 27 anos como congressista, ficou conhecido por seu conservadorismo social e por conflitos públicos,[15] principalmente por ser um vocal opositor dos direitos LGBT e por declarações classificadas como discurso de ódio,[16][17][18] que incluem a defesa das práticas de tortura e assassinatos cometidos pela ditadura militar brasileira,[19] além de falas consideradas racistas.[20] Tido como um político polarizador,[21][22][23] seus pontos de vista e comentários, amplamente descritos como de extrema-direita[24] e populistas,[25] provocaram críticas e também atraíram apoiadores,[26][27][28] gerando o movimento político conhecido como "bolsonarismo".[29][30]

    Sua campanha presidencial foi lançada pelo PSL em agosto de 2018, quando passou a se apresentar como um candidato antissistema,[31] pró-mercado[32] e defensor de valores familiares.[33] Após disputar o segundo turno das eleições de 2018 com Fernando Haddad, do Partido dos Trabalhadores (PT), foi eleito com 55,13% dos votos válidos. Seu governo se caracterizou por forte presença de ministros de formação militar, alinhamento internacional com a direita populista[34][35] e por políticas antiambientais,[36] anti-indigenistas[36] e pró-armas.[37][38] Em 2020, foi nomeado "pessoa do ano" pelo Organized Crime and Corruption Reporting Project por se cercar de figuras corruptas, minar o sistema de justiça e enriquecer grandes proprietários de terras na região amazônica.[39] Foi também responsável por um amplo desmonte das políticas e órgãos da cultura,[40][41] da ciência e da educação,[42][43][44] além de promover repetidos ataques às instituições democráticas[45][46] e fazer maciça divulgação de notícias falsas.[47] Apesar de a criminalidade[48] e o desemprego terem seguido a tendência de queda vista desde o governo Michel Temer,[49] a média de crescimento do PIB foi de cerca de 1,5% ao ano,[50] a precarização do trabalho, a inflação[51] e a fome[52] aumentaram, enquanto a renda per capita, a desigualdade e a pobreza atingiram os piores níveis desde 2012.[53]

    Sua administração envolveu-se em uma série de controvérsias e vários dos ministros que haviam sido indicados originalmente deixaram seus cargos[54] e criticaram o governo.[55] A resposta de Bolsonaro à pandemia de COVID-19 no Brasil também foi reprovada em todo o espectro político e apontada como negacionista,[56][57] depois que ele minimizou os efeitos da doença[58][59] e defendeu tratamentos sem eficácia comprovada, além de ter desestimulado a vacinação,[60][61][62] o uso de máscaras de proteção[63][64] e as medidas de distanciamento social,[65][66][67] posturas que contribuíram para até 400 mil mortes evitáveis[68] e que foram consideradas um crime contra a humanidade pelo Tribunal Permanente dos Povos.[69] Nas eleições de 2022, foi derrotado no segundo turno por Luiz Inácio Lula da Silva (PT), sendo o primeiro presidente do Brasil a não conseguir se reeleger desde a instituição da reeleição em 1997. Tornou-se inelegível após condenação pelo Tribunal Superior Eleitoral (TSE) por abuso de poder político[70] e passou a ser investigado pela Polícia Federal (PF) por suspeitas de crimes contra o patrimônio público e de esquemas de corrupção no Ministério de Educação. Em setembro de 2025, foi condenado a cerca de 27 anos de prisão pelo Supremo Tribunal Federal (STF) pela tentativa de golpe de Estado que envolveu os atos golpistas após as eleições e culminou nos ataques de 8 de janeiro em Brasília.[71][72] A pena começou a ser cumprida como prisão domiciliar em agosto de 2025, mas foi convertida em prisão preventiva em novembro do mesmo ano após Bolsonaro tentar violar a tornozeleira eletrônica. No mesmo mês, a pena definitiva começou a ser cumprida na sede da PF em Brasília. Em janeiro de 2026, foi transferido para a Papudinha, batalhão de tratamento especial do Complexo Penitenciário da Papuda."""

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # llm = ChatGoogleGenerativeAI(
    # llm = ChatVertexAI(
    llm = ChatOllama(
        model="llama3.1",
        temperature=0.3,
        max_retries=1,
    )
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content) 

if __name__ == "__main__":
    main()
    # import google.generativeai as genai
    # genai.configure()

    # for m in genai.list_models():
    #     if "generateContent" in m.supported_generation_methods:
    #         print(m.name)