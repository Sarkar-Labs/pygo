package tensor

import "fmt"

func Fused1252SignMulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_mul_log2: nil operand")
	}
	r := x.Mul(y)
	r = r.Sign()
	r = r.Log2()
	return r, nil
}
func Fused1252SignMulLog2Name() string { return "sign_mul_log2" }
