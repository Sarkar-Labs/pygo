package tensor

import "fmt"

func Fused0872CosMulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_mul_sign: nil operand")
	}
	r := x.Mul(y)
	r = r.Cos()
	r = r.Sign()
	return r, nil
}
func Fused0872CosMulSignName() string { return "cos_mul_sign" }
