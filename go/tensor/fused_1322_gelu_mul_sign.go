package tensor

import "fmt"

func Fused1322GeluMulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_mul_sign: nil operand")
	}
	r := x.Mul(y)
	r = r.GELU()
	r = r.Sign()
	return r, nil
}
func Fused1322GeluMulSignName() string { return "gelu_mul_sign" }
