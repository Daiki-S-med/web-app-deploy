"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import sys
import unicodedata
from dotenv import load_dotenv
import streamlit as st
from docx import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import constants as ct


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理

    これを行わないと、
    ・会話履歴を保存するための変数が用意されていない
    ・ログをみても誰の操作かわからず不具合の調査ができない
    ・Retrieverが用意されていないので、RAGが動かない
    といったことが起こる
    """
    # 初期化データの用意　変数の初期値をセット　情報を入れる箱を用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成　誰がいつアクセスしたか特定できる
    initialize_session_id()
    # ログ出力の設定　アプリの動きを記録
    initialize_logger()
    # RAGのRetriever(検索エンジン)を作成　　
    initialize_retriever()


def initialize_logger():
    """
    ログ出力の設定
    不具合が発生したときに原因特定するために必要
    """
    # 指定のログフォルダが存在すれば読み込み、存在しなければ新規作成
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    
    # 引数に指定した名前のロガー（ログを記録するオブジェクト）を取得
    # 再度別の箇所で呼び出した場合、すでに同じ名前のロガーが存在していれば読み込む
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにロガーにハンドラー（ログの出力先を制御するもの）が設定されている場合、同じログ出力が複数回行われないよう処理を中断する
    # Steremlitが開くたびにロガーがどんどん増えてしまい、
    # 1つの出来事に対して同じログが何重にも書き込まれるのを防ぐための処理
    if logger.hasHandlers():
        return

    # 1日単位でログファイルの中身をリセットし、切り替える設定
    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    # 出力するログメッセージのフォーマット定義
    # %()の中の文字列は、ログを出力するときに自動的に適切な値に置き換えられる

    # - 「levelname」: ログの重要度（INFO, WARNING, ERRORなど）
    # - 「asctime」: ログのタイムスタンプ（いつ記録されたか）
    # - 「lineno」: ログが出力されたファイルの行番号
    # - 「funcName」: ログが出力された関数名
    # - 「session_id」: セッションID（誰のアプリ操作か分かるように） ここは｛｝で
    # - 「message」: ログメッセージ
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )

    # 定義したフォーマッターの適用
    # handler：どこに出力するか決める
    # formatter：どう並べるかを決める
    log_handler.setFormatter(formatter)

    # ログレベルを「INFO」に設定
    # どのレベルから記録されるかを設定　「WARNING」、「ERROR」など
    logger.setLevel(logging.INFO)

    # 作成したハンドラー（ログ出力先を制御するオブジェクト）を、
    # ロガー（ログメッセージを実際に生成するオブジェクト）に追加してログ出力の最終設定
    logger.addHandler(log_handler)


def initialize_session_id():
    """
    セッションIDの作成
    if文を使って、すでにセッションIDが存在する場合は新たに作成しないようにする
    （Streamlitはボタンを押したり入力するたびにプログラムを再実行するため）
    """
    if "session_id" not in st.session_state:
        # ランダムな文字列（セッションID）を、ログ出力用に作成
        # uuid4().hexは、ランダムな16進数の文字列を生成する関数
        st.session_state.session_id = uuid4().hex


def initialize_retriever():
    """
    画面読み込み時にRAGのRetriever（ベクターストアから検索するオブジェクト）を作成
    """
    # ロガーを読み込むことで、後続の処理中に発生したエラーなどがログファイルに記録される
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにRetrieverが作成済みの場合、後続の処理を中断
    if "retriever" in st.session_state:
        return
    
    # RAGの参照先となるデータソースの読み込み
    docs_all = load_data_sources()

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    
    # 埋め込みモデルの用意
    embeddings = OpenAIEmbeddings()
    
    # チャンク分割用のオブジェクトを作成
    text_splitter = CharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP,
        separator="\n"
    )

    # チャンク分割を実施
    splitted_docs = text_splitter.split_documents(docs_all)

    # ベクターストアの作成
    db = Chroma.from_documents(splitted_docs, embedding=embeddings)

    # ベクターストアを検索するRetrieverの作成
    # これでユーザーが質問するたびにチャンク分割をしなくてよくなる
    st.session_state.retriever = db.as_retriever(search_kwargs={"k": ct.RETRIEVER_SEARCH_K})


def initialize_session_state():
    """
    初期化データの用意
    """
    # 初回のときは「messages」がないので、空のリスト[]が作られる
    # 2回目以降(すでに会話が始まっている)場合はスルー
    if "messages" not in st.session_state:
        # 「表示用」の会話ログを順次格納するリストを用意
        st.session_state.messages = []
        # 「LLMとのやりとり用」の会話ログを順次格納するリストを用意
        # AIにこれまでの文脈を伝えるための会話ログ　AIに渡すときはこっちを渡す
        st.session_state.chat_history = []


def load_data_sources():
    """
    RAGの参照先となるデータソースの読み込み
    ファイルから読み込んだデータと、Webページから読み込んだデータの両方を読み込む
    分けて読み込むことで、エラーが起きたときや後からの管理がしやすい

    Returns:
        読み込んだ通常データソース
    """
    # データソースを格納する用のリスト
    docs_all = []
    # ファイル読み込みの実行（渡した各リストにデータが格納される）
    # 指定のフォルダ(ct.RAG_TOP_FOLDER_PATH)内のファイルを再帰的に読み込む関数を呼び出す　
    # フォルダ内のサブフォルダも含めて全部読み込む
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)

    web_docs_all = []
    # ファイルとは別に、指定のWebページ内のデータも読み込み
    # 読み込み対象のWebページ一覧に対して処理
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        # 指定のWebページを読み込み
        loader = WebBaseLoader(web_url)
        web_docs = loader.load()
        # for文の外のリストに読み込んだデータソースを追加
        # 複数のURLがリストにある場合、それらを一つずつ読み込んでリストに追加していく
        web_docs_all.extend(web_docs)
    # 通常読み込みのデータソースにWebページのデータを追加
    docs_all.extend(web_docs_all)

    return docs_all


def recursive_file_check(path, docs_all):
    """
    RAGの参照先となるデータソースの読み込み

    Args:
        path: 読み込み対象のファイル/フォルダのパス
        docs_all: データソースを格納する用のリスト
    """
    # パスがフォルダかどうかを確認
    if os.path.isdir(path):
        # フォルダの場合、フォルダ内のファイル/フォルダ名の一覧を取得
        files = os.listdir(path)
        # 各ファイル/フォルダに対して処理
        for file in files:
            # ファイル/フォルダ名だけでなく、フルパスを取得
            full_path = os.path.join(path, file)
            # フルパスを渡し、再帰的にファイル読み込みの関数を実行
            # ファイルにたどり着くまで操作を繰り返し、たどり着いたらelse以下を実行
            recursive_file_check(full_path, docs_all)
    else:
        # パスがファイルの場合、ファイル読み込み
        file_load(path, docs_all)


def file_load(path, docs_all):
    """
    ファイル内のデータ読み込み

    Args:
        path: ファイルパス
        docs_all: データソースを格納する用のリスト
    """
    # ファイルの拡張子を取得　
    # os.path.splitext()は、ファイルパスを「ファイル名」と「拡張子」に分割する関数
    # 例えば「data.txt」というファイルパスを渡すと、[0]で名前の部分、
    # [1]で「.txt」という拡張子が取得できる
    file_extension = os.path.splitext(path)[1]
    # ファイル名（拡張子を含む）を取得
    file_name = os.path.basename(path)

    # 想定していたファイル形式の場合のみ読み込む　設定ファイル(ct)に用意したリストと照らし合わせる
    if file_extension in ct.SUPPORTED_EXTENSIONS:
        # ファイルの拡張子に合ったdata loaderを使ってデータ読み込み
        loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
        docs = loader.load()
        docs_all.extend(docs)


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    
    Args:
        s: 調整を行う文字列
    
    Returns:
        調整を行った文字列
    """
    # 調整対象は文字列のみ　数字やNoneなどはそのまま返す
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    # Windows環境でRAGを動かすときに、文字コードの問題でエラーが発生することがあるため、
    # 文字列を調整する処理を行う
    # 
    if sys.platform.startswith("win"): # Windows環境であれば
        # Unicode正規化を行う　文字の表現を統一する処理　
        # 例えば「カタカナの『カ』」は、Unicode上では複数の表現方法があるため、これを統一する
        s = unicodedata.normalize('NFC', s)
        # cp932で表現できない文字は無視して進む　エラーを返さないようにする
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    # OSがWindows以外の場合はそのまま返す
    return s