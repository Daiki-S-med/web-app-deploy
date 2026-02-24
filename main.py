"""
このファイルは、Webアプリのメイン処理が記述されたファイルです。
"""

############################################################
# 1. ライブラリの読み込み
############################################################
# 「.env」ファイルから環境変数を読み込むための関数
from dotenv import load_dotenv
# ログ出力を行うためのモジュール
import logging
# streamlitアプリの表示を担当するモジュール
import streamlit as st
# （自作）画面表示以外の様々な関数が定義されているモジュール
import utils
# （自作）アプリ起動時に実行される初期化処理が記述された関数
from initialize import initialize
# （自作）画面表示系の関数が定義されているモジュール
import components as cn
# （自作）変数（定数）がまとめて定義・管理されているモジュール
import constants as ct


############################################################
# 2. 設定関連
############################################################
# ブラウザタブの表示文言を設定
# 「contants.py」で定義された「APP_NAME」変数を参照している
st.set_page_config(
    page_title=ct.APP_NAME
)

# ログ出力を行うためのロガーの設定
# ロガーとは、ファイルへのログ出力を行うためのオブジェクトのこと
# 後続のコード実行中にエラーなどが発生した際、
# ロガーを使ってエラーログをファイルに出力する
logger = logging.getLogger(ct.LOGGER_NAME)


############################################################
# 3. 初期化処理
############################################################
try:
    # 初期化処理（「initialize.py」の「initialize」関数を実行）
    # アプリ起動時に一度だけ実行される
    initialize()
except Exception as e:
    # エラーログの出力
    # 初期化処理の中でエラーが発生した場合、
    # logger.error()でエラーログをファイルに出力
    logger.error(f"{ct.INITIALIZE_ERROR_MESSAGE}\n{e}")
    # st.error()でエラーメッセージの画面表示
    # utils.pyモジュールの「build_error_message」関数を使って、エラーメッセージを装飾
    st.error(utils.build_error_message(ct.INITIALIZE_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    # 後続の処理を中断
    st.stop()

# アプリ起動時のログファイルへの出力
# 「initialized」というキーがセッションステートに存在しない場合、
# つまりアプリ起動時に一度だけ実行される
# エラー以外の単純な情報の記録には「logger.info()」を使用
if not "initialized" in st.session_state:
    st.session_state.initialized = True
    logger.info(ct.APP_BOOT_MESSAGE)


############################################################
# 4. 初期表示
# 画面表示系の関数が定義されているモジュール「components.py」の
# 関数を使って、アプリ起動時の初期表示を行う
############################################################
# タイトル表示
cn.display_app_title()

# サイドバーモードの表示
cn.display_sidebar_content()

# AIメッセージの初期表示
cn.display_initial_ai_message()


############################################################
# 5. 会話ログの表示
############################################################
try:
    # 会話ログの一覧表示
    cn.display_conversation_log()
except Exception as e:
    # エラーログの出力
    # 初期化処理のときと内容は同じ
    logger.error(f"{ct.CONVERSATION_LOG_ERROR_MESSAGE}\n{e}")
    # エラーメッセージの画面表示
    st.error(utils.build_error_message(ct.CONVERSATION_LOG_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    # 後続の処理を中断
    st.stop()


############################################################
# 6. チャット入力の受け付け
############################################################
# 「st.chat_input()」でチャット入力を受け付ける
# アプリ起動時はNoneという値が入っている
chat_message = st.chat_input(ct.CHAT_INPUT_HELPER_TEXT)


############################################################
# 7. チャット送信時の処理　入力されるとTrueとなり処理が実行される
############################################################
if chat_message:
    # ==========================================
    # 7-1. ユーザーメッセージの表示
    # ==========================================
    # ユーザーメッセージのログ出力
    logger.info({"message": chat_message, "application_mode": st.session_state.mode})

    ### ユーザーメッセージを表示
    ### 「ここから先はユーザー側のメッセージとして表示してね」
    with st.chat_message("user"):
        # ユーザーが打った文字を画面上の吹き出しの中に表示
        st.markdown(chat_message)

    # ==========================================
    # 7-2. LLMからの回答取得
    # ==========================================
    # 「st.spinner」でグルグル回っている間、表示の不具合が発生しないよう空のエリアを表示
    # AIの答えが返ってきたら、このemptyの中に答えを入れる
    res_box = st.empty()
    # LLMによる回答生成（回答生成が完了するまでグルグル回す）
    with st.spinner(ct.SPINNER_TEXT):
        try:
            # utils.pyの「get_llm_response」関数を呼び出して、LLMからの回答を取得
            # 画面読み込み時に作成したRetrieverを使い、Chainを実行
            llm_response = utils.get_llm_response(chat_message)
        except Exception as e:
            # エラーログの出力
            logger.error(f"{ct.GET_LLM_RESPONSE_ERROR_MESSAGE}\n{e}")
            # エラーメッセージの画面表示
            st.error(utils.build_error_message(ct.GET_LLM_RESPONSE_ERROR_MESSAGE), icon=ct.ERROR_ICON)
            # 後続の処理を中断
            st.stop()
    
    # ==========================================
    # 7-3. LLMからの回答表示
    # ==========================================

    ### 今回はassistant（AI）側のメッセージとして表示
    ### 画面の左側にAIの吹き出しが表示される
    with st.chat_message("assistant"):
        try:
            # ==========================================
            # モードが「社内文書検索」の場合
            # ==========================================
            if st.session_state.mode == ct.ANSWER_MODE_1:
                # 入力内容と関連性が高い社内文書のありかを表示
                content = cn.display_search_llm_response(llm_response)

            # ==========================================
            # モードが「社内問い合わせ」の場合
            # ==========================================
            elif st.session_state.mode == ct.ANSWER_MODE_2:
                # 入力に対しての回答と、参照した文書のありかを表示
                content = cn.display_contact_llm_response(llm_response)
            
            # AIメッセージのログ出力
            logger.info({"message": content, "application_mode": st.session_state.mode})
        except Exception as e:
            # エラーログの出力
            logger.error(f"{ct.DISP_ANSWER_ERROR_MESSAGE}\n{e}")
            # エラーメッセージの画面表示
            st.error(utils.build_error_message(ct.DISP_ANSWER_ERROR_MESSAGE), icon=ct.ERROR_ICON)
            # 後続の処理を中断
            st.stop()

    # ==========================================
    # 7-4. 会話ログへの追加
    # ==========================================
    # 表示用の会話ログにユーザーメッセージを追加
    st.session_state.messages.append({"role": "user", "content": chat_message})
    # 表示用の会話ログにAIメッセージを追加
    st.session_state.messages.append({"role": "assistant", "content": content})