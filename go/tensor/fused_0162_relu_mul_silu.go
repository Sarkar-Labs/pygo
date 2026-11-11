package tensor

import "fmt"

func Fused0162ReluMulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_mul_silu: nil operand")
	}
	r := x.Mul(y)
	r = r.ReLU()
	r = r.SiLU()
	return r, nil
}
func Fused0162ReluMulSiluName() string { return "relu_mul_silu" }
