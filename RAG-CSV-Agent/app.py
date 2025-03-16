import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
import base64
import math
import os
import asyncio
import aiohttp
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai

# APIキーを直接設定
OPENAI_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyBVi8qwD4lKjJ7IL76S_edr7Scmsl9mFZ8"

# Webhook設定を直接コードに埋め込む
enable_webhook = True
webhook_url = "https://hook.us1.make.com/yv19q1vynuxr36v9i663ygk3e2eqv4it"

# Google Gemini API設定
genai.configure(api_key=GEMINI_API_KEY)

def get_download_link(df_chunks):
    """Generate a download link for a zip file containing all chunks"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as zip_file:
        for i, chunk in enumerate(df_chunks):
            csv_buffer = io.StringIO()
            chunk.to_csv(csv_buffer, index=False, header=True)
            zip_file.writestr(f"chunk_{i+1}.csv", csv_buffer.getvalue())
    
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    return f'<a href="data:application/zip;base64,{b64}" download="csv_chunks.zip">ダウンロード: 分割されたCSVファイル</a>'

def calculate_chunks_needed(total_rows, chunk_size=20):
    """指定した行数と分割サイズから必要なチャンク数を計算"""
    return math.ceil(total_rows / chunk_size)

def split_csv_into_chunks(df, chunk_size=20):
    """Split a DataFrame into chunks of specified row size"""
    total_rows = len(df)
    chunks = []
    
    for i in range(0, total_rows, chunk_size):
        end_idx = min(i + chunk_size, total_rows)
        chunk = df.iloc[i:end_idx, :].copy()
        chunks.append(chunk)
    
    return chunks

async def call_openai_api(session, chunk_df, task_type, api_key):
    """非同期でOpenAI APIを呼び出す"""
    # CSVデータを文字列に変換
    csv_content = chunk_df.to_csv(index=False)
    columns_list = chunk_df.columns.tolist()
    
    # タスクに応じたプロンプトを生成
    if task_type == "sales_analysis":
        system_prompt = f"""
        あなたは与えられたCSVデータから販売分析レポートを作成する専門家です。
        データを分析して、以下の内容を含む詳細な分析レポートを作成してください：
        1. 全体的な売上傾向
        2. 主要な指標の概要
        3. データから得られる重要な洞察
        4. 推奨事項（もしあれば）
        
        重要: 分析では全ての列({len(columns_list)}列)を考慮してください。
        ただし、レポート内のテーブルには以下の3つの列のみを表示してください：
        1. 商品名（製品名や品目名などの商品を識別する列）
        2. 型番（製品コードや商品IDなどの識別子）
        3. 値段（価格や売上金額など）
        
        テーブル表示は上記3つの列のみに限定し、他の列のデータは文章での分析に使用してください。
        表示する列が見つからない場合は、最も近いと思われる列を選択してください。
        
        CSVデータには必要な情報が含まれています。返答はマークダウン形式で、
        見出しや箇条書きを使用して読みやすくしてください。
        """
    elif task_type == "ad_analysis":
        system_prompt = f"""
        あなたは与えられたCSVデータから広告パフォーマンス分析レポートを作成する専門家です。
        データを分析して、以下の内容を含む詳細な分析レポートを作成してください：
        1. 広告キャンペーンの全体的なパフォーマンス
        2. ROI、ROAS、CPAなどの主要指標の分析
        3. 最も効果的な広告チャネルや広告タイプの特定
        4. コンバージョン率の分析
        5. 費用対効果の高い広告施策の提案
        
        重要: 分析では全ての列({len(columns_list)}列)を考慮してください。
        ただし、レポート内のテーブルには以下の3つの列のみを表示してください：
        1. 広告名/チャネル名（広告やキャンペーンを識別する列）
        2. 広告ID/コード（キャンペーンの識別子）
        3. パフォーマンス指標（コスト、ROI、コンバージョン率など）
        
        テーブル表示は上記3つの列のみに限定し、他の列のデータは文章での分析に使用してください。
        表示する列が見つからない場合は、データの内容から最も近いと思われる列を選択してください。
        
        CSVデータには必要な情報が含まれています。返答はマークダウン形式で、
        見出しや箇条書きを使用して読みやすくしてください。
        効果的な広告戦略のための具体的な改善提案も含めてください。
        """
    elif task_type == "review_analysis":
        system_prompt = f"""
        あなたは、ECサイトに特化したデータアナリスト兼コンサルタントです。ユーザーから提出される各種資料を解析し、仕入候補商品の自動提案、最適な販売戦略の策定、そして既存製品のレビューから改善点を抽出するツールとして動作してください。さらに、毎日の販売データや広告データ、セラースプライトデータを継続的に学習し、システムの精度を向上させる機能も有します。

        【Step 1: データ取得および初期学習】

        ユーザーが添付する資料から、以下の項目を自動抽出し、各商品の基本情報として初期学習モデルに取り込みます。

        商品名情報

        → タイトル（必要に応じて、ブランドも参照）

        競合売上と利益推定

        → 価格(￥)と月間販売数と月間販売額(￥)とFBA(￥)と粗利益と月間販売数成長率

        ランキング情報

        →カテゴリーと親カテゴリーと親カテゴリーBSRと親カテゴリーBSR増加数と親カテゴリーBSR成長率と子カテゴリーと子カテゴリーBSR

        レビュー情報

        → 評価数、評価値、評価率、および必要に応じて Q&A

        その他、対象期間、対象プラットフォーム、各商品の基本データを取り込み、目標利益率など既定ルールもシステムに組み込みます。

        Step1完了後、次のステップ（Step2）へ進む旨をユーザーに通知してください。

        【Step 2: 利益予測シミュレーション】

        各候補商品について、仕入れおよび販売に関わるコストと収益をもとに、利益予測を行います。

        想定売価

        → 価格(￥)と粗利益

        コスト項目

        → 自社の仕入れコスト、輸送費、FBA手数料（FBA(￥)）、広告費などの内部データを活用

        → ただし、これらの内部コストデータが提供されていない場合は、過去実績や業界平均に基づく標準値を自動算出するモデルを適用してください。

        - 例：各コスト項目に対して、既存の類似商品や市場ベンチマークから平均値を導出し、シミュレーションに用いる

        - レポート上では「内部コストは推定値を用いて計算しています」と明記すること

        シミュレーション計算

        → 予想粗利 = 想定売価 − (仕入れコスト + 輸送費 + FBA手数料 + 広告費)

        → 予測利益率を算出し、利益が最大化される商品を特定

        シミュレーション結果を基に、各商品の仕入れ可能性および最適な販売戦略を評価します。

        Step2完了後、Step3への進行と追加資料の添付をユーザーに促してください。

        【Step 3: 競合レビュー分析と機能提案】

        添付資料に含まれる競合他社のレビュー情報を解析し、製品の強み・弱みやユーザーの求める機能を明らかにします。

        テキスト情報

        → タイトル、製品概要（製品の特徴や差別化ポイントを把握するため）

        レビュー定量情報

        → 評価数、評価値、評価率

        補助情報

        → Q&A（ユーザーの具体的な意見や疑問点を抽出）

        これにより、各商品の競争力向上のために必要な「マスト機能」および「追加機能」を自動提案します。

        Step3完了後、Step4への進行と必要な追加資料の添付をユーザーに促してください。

        【Step 4: 既存製品レビュー分析と改善提案】

        自社の既存製品に対するレビュー資料から、改善が必要な点を詳細に解析します。対象となる資料は以下の項目を含みます。

        ASIN

        → 製品識別用

        タイトル

        → レビューの見出しや概要確認用

        内容

        → レビュー本文（具体的な改善点やフィードバックを抽出）

        認証購入者レビュー

        → 信頼性の高いフィードバックとして重視

        型番

        → 製品のバリエーション確認用

        星評価

        → 定量的評価の把握

        役に立つ数

        → ユーザー評価の信頼度指標

        レビュー時間

        → 時系列での傾向分析

        解析結果に基づき、既存製品の改善策や次のアクションプランを提示してください。

        Step4完了後、次のステップ（最終レポート生成など）への進行をユーザーに通知してください。

        【最終出力】

        上記のStep1～Step4に基づき、仕入候補商品の提案、シミュレーション結果、競合レビューおよび既存製品レビューの分析結果を統合したレポートを自動生成してください。レポートは以下の構成とします。

        タイトル

        対象期間、プラットフォーム名

        表（仕入候補商品のランキング：予想粗利・利益率の高い商品トップ10、各商品のシミュレーション結果【想定売価、推定仕入れコスト、輸送費、FBA手数料、広告費、予想粗利】）

        詳細なレポート（仕入提案、競合レビュー分析結果、改善提案など）

        データに基づく主要トピックの抽出と、その詳細レポート（箇条書きと本文を入れた2000文字以上、改善策および次のアクションプランを含む）

        [箇条書き]

        [本文]

        【フォールバック対応】

        資料に必要なセラースプライトデータやレビュー情報が不足している場合は、出力内に「○○の情報が不足しています。追加資料の提供をお願いします。」と明記してください。
        
        これは複数のチャンクに分割され、各チャンクごとに分析が行われています。
        以下のOpenAIによる分析結果を統合し、このレポート構成に従って包括的なレポートを作成してください。
        """
    else:
        system_prompt = f"""
        与えられたCSVデータを分析し、有用な洞察を提供してください。
        
        重要: 分析では全ての列({len(columns_list)}列)を考慮してください。
        ただし、結果のテーブルには以下の3つの列のみを表示してください：
        1. 商品名（製品名や品目名などの商品を識別する列）
        2. 型番（製品コードや商品IDなどの識別子）
        3. 値段（価格や売上金額など）
        """
    
    user_prompt = f"""以下のCSVデータを分析してください。
    
分析には全ての列({len(columns_list)}列)を使用してください。
ただし、テーブル表示は「商品名」「型番」「値段」の3つの列のみに限定してください。
これらの列が正確に一致しない場合は、データの内容から最も近いと思われる列を選んでください。

{csv_content}"""
    
    # APIリクエストを準備
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_completion_tokens": 10000
    }
    
    # 非同期でAPIリクエストを送信
    try:
        async with session.post("https://api.openai.com/v1/chat/completions", 
                               headers=headers, 
                               json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result["choices"][0]["message"]["content"]
            else:
                error_text = await response.text()
                return f"APIエラー (ステータスコード: {response.status}): {error_text}"
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"

async def process_chunks_with_openai(chunks, task_type, api_key):
    """複数のチャンクを非同期で処理"""
    async with aiohttp.ClientSession() as session:
        tasks = [call_openai_api(session, chunk, task_type, api_key) for chunk in chunks]
        return await asyncio.gather(*tasks)

def run_openai_analysis(chunks, task_type, api_key):
    """OpenAI分析を実行するためのメインエントリポイント"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(process_chunks_with_openai(chunks, task_type, api_key))
    return results

def generate_summary_with_gemini(openai_results, original_df_info, analysis_type, file_name=None):
    """Gemini Flash Thinkingを使用して、OpenAIの分析結果を総括する"""
    try:
        if not GEMINI_API_KEY:
            return "Gemini APIキーが設定されていないため、総括分析を実行できません。「APIキーの設定について」セクションでGemini APIキーを設定してください。"
        
        # Geminiモデルの設定
        model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21', generation_config={"max_output_tokens": 60000})
        
        # 元データの情報
        rows, cols = original_df_info
        
        # 分析タイプに応じたプロンプト生成
        if analysis_type == "広告分析レポート":
            system_prompt = f"""
            あなたはECサイトに特化したデータアナリストでありコンサルタントです。  
            ユーザーから提出された広告レポート資料（csv、pdf、Excelなど）をもとに、以下の手順に従って自動的に広告運用分析レポートを生成してください。  
            なお、ユーザーからの追加入力はなく、すべて添付資料および既定ルールに基づいて処理を実施し、最終出力としてコンテンツのみを返してください。  
            必要なデータが不足している場合は、出力内に「○○の情報が不足しています。追加資料の提供をお願いします。」と記載してください。

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            ■ 前提条件
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            ・添付資料には、対象期間（開始日、終了日）、対象プラットフォーム、各キャンペーンの基本データが含まれている。  
            ・抽出する必須データは以下の通り：  
            　- 開始日、終了日  
            　- キャンペーン名  
            　- ターゲティング  
            　- マッチタイプ  
            　- カスタマーの検索キーワード  
            　- インプレッション  
            　- クリック  
            　- クリックスルー率 (CTR)  
            　- 平均クリック単価 (CPC)  
            　- 費用（広告費）  
            　- 広告がクリックされてから7日間の総売上高  
            　- 広告費売上高比率（ACOS）合計  
            　- 広告費用対効果（ROAS）合計  
            　- 広告がクリックされてから7日間の合計注文数  
            　- 広告がクリックされてから7日間の合計販売数  
            ・既定ルールとして、目標ACOS（例：20%以下）や目標ROAS（例：5.0以上）はシステム内に設定済みとする。

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            ■ Step 1: データ取得
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            添付資料および関連システムから、以下のデータを自動抽出する：
            　・ 開始日、終了日  
            　・ キャンペーン名  
            　・ ターゲティング  
            　・ マッチタイプ  
            　・ カスタマーの検索キーワード  
            　・ インプレッション、クリック、CTR、平均クリック単価、費用  
            　・ 広告がクリックされてから7日間の総売上高  
            　・ ACOS合計、ROAS合計  
            　・ 広告がクリックされてから7日間の合計注文数、合計販売数

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            ■ Step 2: 分析計算
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            各キャンペーンについて、以下を計算する：
            　1. 実績ACOS = (費用 ÷ 広告がクリックされてから7日間の総売上高) × 100  
            　2. 実績ROAS = 広告がクリックされてから7日間の総売上高 ÷ 費用  
            　3. 目標ACOSやROASとの乖離を算出し、最適なクリック単価、デイリーバジェットの変更案、必要に応じた売価変更および出稿停止のシミュレーションを実施する。

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            ■ Step 3: シミュレーション・提案
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            各キャンペーンごとに、目標ACOS/ROASに合わせた最適な出稿単価や広告予算の変更案を算出する。  
            また、売上や在庫状況（他システムと連動している場合）に基づき、売価変更または一時的な出稿停止の提案も自動算出する。

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            ■ Step 4: レポート生成
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            以下の【レポート構成】に従い、最終的な広告運用分析レポートのコンテンツのみを出力する。あなたは、20年以上の経験を持つ百戦錬磨のEC売上向上コンサルタントなので、クライアントに提示するレポートとして仕上げて下さい。余計なメタ情報は含まず、コンテンツのみとする。

            【レポート構成】
            以下の構成に従って、広告運用分析レポートの最終出力を生成してください。最終出力は必ず以下の各セクションを含むこと。余計なメタ情報は一切含まず、コンテンツのみを出力してください。

            1. 【レポートタイトル】
               - 例：「[広告運用分析レポート] 2024年12月08日～2025年02月10日（対象プラットフォーム）の広告最適化提案」

            2. 【期間・プラットフォーム情報】
               - 分析対象の開始日、終了日、および対象プラットフォーム（例：Amazon.co.jp）を明記する。

            3. 【表形式の集計結果】
               a. 広告費用・売上実績ランキング（トップ10）
                  - 表に含める項目：キャンペーン名、ターゲティング、マッチタイプ、カスタマーの検索キーワード、インプレッション、クリック、CTR、平均クリック単価、費用、総売上高（7日間）、実績ACOS、実績ROAS、合計注文数、合計販売数
               b. 日別広告効果推移
                  - 表に含める項目：日付、売上金額、注文数、最高売上日、最低売上日（またはその他の指標）
               c. キャンペーン別詳細データ
                  - 各キャンペーンの主要指標をまとめた表

            4. 【全体の詳細な統括レポート】
               - 1.総広告費、総売上、全体ACOS、全体ROASなど、全体パフォーマンスの総括情報を詳細に記載する。
               -2.1の内容や全体のデータを鑑みて、かなり詳細でマーケ初心者でも分かりやすく噛み砕いた内容のレポートを1500文字以上で作成。

            5. 【トピック抽出（表形式）】
               - データから抽出された注目すべきキャンペーンやキーワードの情報を、例えば「高ACOSキャンペーン」「CTRは高いがコンバージョンが低いキーワード」など、表形式で提示する。

            6. 【詳細な分析・考察レポート】
               - 箇条書きと本文の構成で少なくとも1000文字以上の詳細なテキスト分析を記載する。  
               - 各キャンペーンやキーワードのパフォーマンスに基づく考察、改善点、具体的なアクションプラン（例：クリック単価の変更、予算調整、売価変更、出稿停止など）を盛り込み、説得力のある内容とする。

            7. 【まとめ】
               - 全体の結論と、今後の広告運用戦略に関するアクションアイテムを明確に記載する。

            【注意事項】
            - 出力時は、上記各セクションのタイトルとその内容を必ず含め、セクションごとに区切り線や見出しで整理してください。
            - もし添付資料内の必要な情報が不足している場合は、出力内に「○○の情報が不足しています。追加資料の提供をお願いします。」と記載してください。
            - ユーザーからの追加入力は一切なく、添付資料と既定ルールに基づき自動処理してください。
            - 出力サイズが1回のレポートで限界を迎える場合、2回に分けてレポートを作成して良いので、クオリティがかなり詳細で分かりやすいレポートを作成することを意識してください。
            - データの量は関係なく、1回のレポートで完結するようにしてください。
            - 

            元データは全部で{rows}行×{cols}列のCSVファイルです。
            これは複数のチャンクに分割され、各チャンクごとに分析が行われています。
            以下のOpenAIによる分析結果を統合し、このレポート構成に従って包括的なレポートを作成してください。
            """
        elif analysis_type == "レビュー分析レポート":
            system_prompt = f"""
            あなたはデータ分析の専門家です。OpenAIによって生成された複数の顧客レビュー・感想分析レポートを総括して、
            包括的な最終分析レポートを作成してください。

            以下のポイントに注目してください：
            1. 複数のチャンクの分析結果を統合し、全体的な顧客満足度と感情分析を把握する
            2. 最も頻繁に言及されているキーワードやトピックを特定する
            3. 製品/サービスの強みと改善が必要な点を明確にする
            4. 競合他社や類似製品との比較分析（データに含まれている場合）
            5. 顧客セグメント別の傾向分析
            6. 時系列での顧客フィードバックの変化（データに含まれている場合）

            元データは全部で{rows}行×{cols}列のCSVファイルです。
            これは複数のチャンクに分割され、各チャンクごとに分析が行われています。
            
            最終レポートには以下のセクションを含めてください：
            - エグゼクティブサマリー
            - 感情分析の概要（ポジティブ/ネガティブ/中立の割合）
            - 主要なキーワードとトピックの分析
            - 製品/サービスの強みと改善点
            - 顧客満足度向上のための具体的な推奨事項
            - 競合分析（データに含まれている場合）
            
            テーブルには製品/サービス名、評価/スコア、キーワード/感想の情報を明確に示してください。
            顧客体験を向上させるための実用的かつ具体的な改善策を提案してください。
            """
        else:  # 販売分析レポート
            system_prompt = f"""
            あなたはECサイトに特化したデータアナリストでありコンサルタントです。ユーザーから提出された資料（csv、pdf、Excelなど）を下記の通り遂行してください。

            Step 1: データ取得
            ユーザーが資料を添付するので、添付資料から、各商品の以下のデータを自動抽出する:
            「商品名ｌ売上数量ｌ売上金額ｌ仕入れコストｌ送料ｌ代引手数料ｌ消費税ｌ税込合計ｌFBA手数料ｌ広告費ｌ在庫数」
            添付資料には、対象期間、対象モール、各商品の基本データが含まれている。
            リードタイムや在庫条件（例: FBA倉庫に最低50個を確保するなど）は資料内に記載、または既定ルールとしてシステムに組み込む。

            Step 2: 分析計算
            各商品について以下を計算する:
            純利益 = 売上金額 − (各種経費)
            在庫回転率 = 販売数量 ÷ (期間中の平均在庫数)
            商品ランク（ABC分析） = 売上金額や純利益の大小に基づき、A/B/C に分類

            Step 3: 分析計算
            過去の販売実績と現在の在庫数、既定のリードタイムをもとに、各商品の今後の予測販売数を算出し、
            推奨発注数を計算する。
            必要に応じ、FBA倉庫、国内自社倉庫、委託倉庫への振り分け数も自動提案する。

            Step 4: レポート生成
            最終出力として、以下の構成に従いレポートを作成する。あなたは、20年以上の経験を持つ百戦錬磨のEC売上向上コンサルタントなので、クライアントに提示するレポートとして仕上げて下さい。
            【構成】
            -タイトル
            -期間、プラットフォーム名
            -表（
            🔹 売上ランキング（売上金額の高い商品トップ10）
            🔹 販売個数ランキング（販売数が多い商品トップ10）
            🔹 日別売上推移（売上が多かった日、少なかった日）
            🔹 商品ごとの売上集計（SKUごとの売上データ））
            -全体の詳細なレポート
            -データを基にトピックとなる部分を表として抜き取り表示
            -トピックの詳細なレポート（1000文字以上）

            元データは全部で{rows}行×{cols}列のCSVファイルです。
            これは複数のチャンクに分割され、各チャンクごとに分析が行われています。
            以下のOpenAIによる分析結果を統合し、このレポート構成に従って包括的なレポートを作成してください。
            
            ※必要なデータが不足している場合は、「○○の情報が不足しています。追加資料の提供をお願いします。」と簡潔に報告してください。
            """
        
        # 各チャンクの分析結果を結合
        combined_analyses = ""
        for i, result in enumerate(openai_results):
            combined_analyses += f"\n---- チャンク{i+1}の分析結果 ----\n{result}\n"
        
        # Geminiへのプロンプト
        prompt = f"""
        {system_prompt}

        以下はOpenAIによる各チャンクの分析結果です：
        {combined_analyses}

        これらの分析結果を統合して、包括的な最終分析レポートを作成してください。
        見やすいマークダウン形式で、適切な見出し、箇条書き、テーブルを使用してください。
        """
        
        # Geminiで総括分析を実行
        response = model.generate_content(prompt)
        summary_text = response.text
        
        # Make.com Webhookにデータを送信
        if enable_webhook and webhook_url:
            try:
                # 送信データの作成
                webhook_data = {
                    "analysis_type": analysis_type,
                    "summary": summary_text,
                    "data_info": {
                        "total_rows": rows,
                        "total_columns": cols,
                        "chunks_analyzed": len(openai_results),
                        "file_name": file_name if file_name else "unknown_file"
                    },
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "analysis_metadata": {
                        "model": "gemini-2.0-pro-exp-02-05",
                        "openai_model": "gpt-4o-mini",
                        "chunks_size": chunk_size
                    }
                }
                
                # Webhookへデータ送信
                webhook_response = requests.post(webhook_url, json=webhook_data)
                
                # セッション状態にステータスを保存
                if webhook_response.status_code == 200:
                    st.session_state.webhook_status = f"✅ 分析結果がMake.comに正常に送信されました（ステータス: {webhook_response.status_code}）"
                else:
                    st.session_state.webhook_status = f"❌ Make.comへの送信が失敗しました: {webhook_response.status_code}"
            except Exception as e:
                st.session_state.webhook_status = f"❌ Webhookへの送信中にエラーが発生しました: {str(e)}"
        
        # 結果を返す
        return summary_text
    
    except Exception as e:
        return f"Geminiによる総括分析中にエラーが発生しました: {str(e)}"

# Streamlitアプリの初期化
if 'webhook_status' not in st.session_state:
    st.session_state.webhook_status = ""

st.title("CSVファイル分割ツール & LLM分析")
st.write("CSVファイルを20行ごとのチャンクに分割し、OpenAIによる分析を実行します")

# サイドバー設定
st.sidebar.write("## 設定")
chunk_size = 20
custom_chunk_size = st.sidebar.number_input("チャンクサイズ（行数）", 
                                          min_value=1, 
                                          max_value=100, 
                                          value=chunk_size,
                                          step=1)
if custom_chunk_size != chunk_size:
    chunk_size = custom_chunk_size
    st.sidebar.success(f"チャンクサイズを {chunk_size} 行に設定しました")

# APIキーはもうサイドバーで入力不要
api_key = OPENAI_API_KEY

# ファイルアップロード
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        total_rows = len(df)
        st.write(f"アップロードされたCSVファイル: {uploaded_file.name}")
        st.write(f"元のサイズ: {df.shape[0]}行 × {df.shape[1]}列")
        
        chunks_needed = calculate_chunks_needed(total_rows, chunk_size)
        st.write(f"分割予測: {total_rows}行を{chunk_size}行ずつ分割すると、{chunks_needed}個のチャンクになります")
        
        example_chunks = calculate_chunks_needed(700, chunk_size)
        st.write(f"参考: 700行のCSVファイルは{chunk_size}行ずつ分割すると、{example_chunks}個のチャンクになります")
        
        # データサンプル表示
        st.write("データのサンプル:")
        st.dataframe(df.head())
        
        # チャンク分割
        chunks = split_csv_into_chunks(df, chunk_size)
        st.write(f"分割結果: {len(chunks)}チャンク (各チャンクには全ての列と対応する行が含まれます)")
        
        for i, chunk in enumerate(chunks):
            st.write(f"チャンク {i+1}: {chunk.shape[0]}行 × {chunk.shape[1]}列")
        
        # チャンクプレビュー
        chunk_to_view = st.selectbox("プレビューするチャンクを選択:", 
                                   range(1, len(chunks) + 1),
                                   format_func=lambda x: f"チャンク {x}")
        if chunk_to_view:
            st.write(f"チャンク {chunk_to_view} のプレビュー:")
            st.dataframe(chunks[chunk_to_view-1].head())
        
        # CSVダウンロードリンク
        st.markdown(get_download_link(chunks), unsafe_allow_html=True)
        
        # LLM分析セクション
        st.write("---")
        st.write("## OpenAI LLM分析")
        st.write("分割したCSVチャンクを使用してLLM分析を実行できます。テーブルには商品名、型番、値段の3つの列のみが表示されます。")
        
        # 分析タイプ選択
        analysis_type = st.radio(
            "分析タイプを選択してください:",
            ["販売分析レポート", "広告分析レポート", "レビュー分析レポート"]
        )
        
        # チャンク数制限（処理負荷とコスト削減のため）
        max_chunks_for_analysis = min(5, len(chunks))
        chunks_for_analysis = st.slider(
            "分析に使用するチャンク数を選択 (コストと処理時間を考慮して制限してください):",
            1, len(chunks), min(3, len(chunks))
        )
        
        if st.button("LLM分析を実行"):
            if not api_key:
                st.error("OpenAI APIキーが設定されていません。サイドバーでAPIキーを入力してください。")
            else:
                if analysis_type == "広告分析レポート":
                    task_type = "ad_analysis"
                elif analysis_type == "レビュー分析レポート":
                    task_type = "review_analysis"
                else:
                    task_type = "sales_analysis"
                
                # プログレスバーの表示
                progress = st.progress(0)
                status_text = st.empty()
                
                status_text.text("分析の準備中...")
                time.sleep(1)
                
                # 分析用のチャンクを準備（選択された数だけ）
                selected_chunks = chunks[:chunks_for_analysis]
                
                # 分析開始
                status_text.text("LLMによる分析を実行中...")
                
                try:
                    # 並列処理で分析を実行
                    results = run_openai_analysis(selected_chunks, task_type, api_key)
                    
                    # 結果表示
                    status_text.text("分析完了！結果を表示します")
                    progress.progress(100)
                    
                    # タブでチャンクごとの分析結果を表示
                    st.write(f"### {analysis_type}の結果")
                    tabs = st.tabs([f"チャンク {i+1}" for i in range(len(selected_chunks))])
                    
                    for i, (tab, result) in enumerate(zip(tabs, results)):
                        with tab:
                            st.markdown(result)
                    
                    # すべての結果を結合した総合分析も表示
                    with st.expander("すべてのチャンクの分析結果一覧"):
                        combined_results = "\n\n".join([f"**チャンク {i+1}の分析**\n{res}" for i, res in enumerate(results)])
                        st.markdown(combined_results)
                    
                    # Geminiによる総括分析の実行
                    st.write("---")
                    st.write("## Geminiによる総括分析")
                    
                    with st.spinner("Gemini AIによる総括分析を実行中..."):
                        # 元データの情報
                        original_df_info = (df.shape[0], df.shape[1])
                        
                        # Webhookステータス表示用
                        webhook_status = st.empty()
                        if enable_webhook:
                            webhook_status.info("分析結果をMake.comに自動送信されます")
                        
                        # ファイル名を取得
                        file_name = uploaded_file.name if uploaded_file else None
                        
                        # Geminiによる総括分析を実行
                        summary = generate_summary_with_gemini(results, original_df_info, analysis_type, file_name)
                        
                        # Webhook送信結果の表示
                        if enable_webhook:
                            if "正常に送信" in st.session_state.get('webhook_status', ''):
                                webhook_status.success(st.session_state.webhook_status)
                            elif "失敗" in st.session_state.get('webhook_status', ''):
                                webhook_status.error(st.session_state.webhook_status)
                            else:
                                webhook_status.empty()
                        
                        # 結果を表示
                        st.markdown(summary)
                    
                except Exception as e:
                    st.error(f"分析中にエラーが発生しました: {str(e)}")
                    
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")