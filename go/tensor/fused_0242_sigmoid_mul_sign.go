package tensor

import "fmt"

func Fused0242SigmoidMulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_mul_sign: nil operand")
	}
	r := x.Mul(y)
	r = r.Sigmoid()
	r = r.Sign()
	return r, nil
}
func Fused0242SigmoidMulSignName() string { return "sigmoid_mul_sign" }
