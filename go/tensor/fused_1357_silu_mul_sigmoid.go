package tensor

import "fmt"

func Fused1357SiluMulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_mul_sigmoid: nil operand")
	}
	r := x.Mul(y)
	r = r.SiLU()
	r = r.Sigmoid()
	return r, nil
}
func Fused1357SiluMulSigmoidName() string { return "silu_mul_sigmoid" }
