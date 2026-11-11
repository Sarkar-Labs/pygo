package tensor

import "fmt"

func Fused1202SignMulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_mul_log: nil operand")
	}
	r := x.Mul(y)
	r = r.Sign()
	r = r.Log()
	return r, nil
}
func Fused1202SignMulLogName() string { return "sign_mul_log" }
