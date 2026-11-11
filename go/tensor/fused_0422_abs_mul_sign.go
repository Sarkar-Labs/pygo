package tensor

import "fmt"

func Fused0422AbsMulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_mul_sign: nil operand")
	}
	r := x.Mul(y)
	r = r.Abs()
	r = r.Sign()
	return r, nil
}
func Fused0422AbsMulSignName() string { return "abs_mul_sign" }
