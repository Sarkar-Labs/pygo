package tensor

import "fmt"

func Fused0602ExpMulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_sign: nil operand")
	}
	r := x.Mul(y)
	r = r.Exp()
	r = r.Sign()
	return r, nil
}
func Fused0602ExpMulSignName() string { return "exp_mul_sign" }
