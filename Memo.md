# Note
### My words
- 取樣是把聲音這種類比訊號數位化時的方法，取樣率表示要以每秒多少次去記錄聲音，而根據取樣定律，當取樣率為 44.1kHz 時，所能記錄下來的最高聲音頻率為 22050Hz，這已經涵蓋了人類耳朵能聽到的頻率範圍 20Hz ~ 20kHz。當我們要將錄製下來的音樂檔案還原為？時，也應當使用同樣的取樣率 44.1kHz。[1]
[1] https://ysolife.com/recording-sampling-rate/

### Not my words
- MFCC 最後產生的特徵是 39 維，包含12個倒頻譜係數加上對數能量(log energy)，以及這13個向量的一階和二階(delta and delta-delta)導數。[1]
[1] https://ithelp.ithome.com.tw/articles/10267054

### Just some little tips
- epoch 唸 epo 就好


# To-do
## SR
- 目標是 7成
- 先確定 Librosa 回傳的 MFCC 特徵值的 20 是什麼內容（可能要去看原始碼才能知道）
- MFCC 特徵取(13, 104) or (26, 104) or (39, 104) 去做實驗，不要取最長的，取已知音檔內容的長度（約不到一秒），轉出的第二個特徵長度是 104，因此固定以這個長度去做截斷與補零
- 以請開、請關、請保持這三個類別做預測
## YT
- 找一個YouTube直播的頻道去思考要抓什麼資料