#include <string>
#include <iostream>
using namespace std;
/**
凯撒密码类（面对对象封装）
 **/
class CaesarCiper{
private:
    int shift;//原始密钥，偏移量（正数右移 负数左移）
    /*
    规范偏移量到0-25
    */
    int normalizeShift(int rawShift) const {
        int n = rawShift % 26;
        if (n < 0) n = n+26;
        return n;
    }
    /*
    核心：偏移实现
    */
   char shiftChar(char ch, int nShift) const {
    if (ch >= 'A' && ch <= 'Z'){
        return static_cast<char>((ch - 'A' + nShift) % 26 + 'A');
    } else if (ch >= 'a' && ch <= 'z'){
        return static_cast<char>((ch - 'a' + nShift)% 26 + 'a');
    }
    return ch;
   }

public:
   /*
   构造函数 获取密钥
   */
  explicit CaesarCiper(int key=0): shift(key){}

  void setShift(int key){
    shift = key;
  }
  int getShift()const {
    return shift;
  }

/*
  加解密实现
*/
  string encrypt(const string & plaintext) const {
    string result = plaintext;
    int nShift = normalizeShift(shift);
    for (char &ch : result){
        ch =shiftChar(ch,nShift);
    }
    return result;
}
  string decrypt(const string & ciphertext) const {
    string result = ciphertext;
    int nShift = normalizeShift(-shift);
    for (char &ch : result){
        ch = shiftChar(ch,nShift);
    }
    return result;
  }
};

int main(){
    int n;
    cout << "请输入一个数字作为密钥：" << endl;
    cin >> n ;
    CaesarCiper cipher(n);
    string plain = "Hello,world! 2026";
    string encrypted = cipher.encrypt(plain);
    string decrypted = cipher.decrypt(encrypted);
    cout << "原文 ：" << plain << endl;
    cout << "密文 ：" << encrypted << endl;
    cout << "解密 ：" << decrypted << endl;

    return 0;
}