package tensor

import "fmt"

func Fused0542ExpMulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_relu: nil operand")
	}
	r := x.Mul(y)
	r = r.Exp()
	r = r.ReLU()
	return r, nil
}
func Fused0542ExpMulReluName() string { return "exp_mul_relu" }
