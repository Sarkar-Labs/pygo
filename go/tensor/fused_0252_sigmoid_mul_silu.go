package tensor

import "fmt"

func Fused0252SigmoidMulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_mul_silu: nil operand")
	}
	r := x.Mul(y)
	r = r.Sigmoid()
	r = r.SiLU()
	return r, nil
}
func Fused0252SigmoidMulSiluName() string { return "sigmoid_mul_silu" }
