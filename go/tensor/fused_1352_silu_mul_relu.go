package tensor

import "fmt"

func Fused1352SiluMulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_mul_relu: nil operand")
	}
	r := x.Mul(y)
	r = r.SiLU()
	r = r.ReLU()
	return r, nil
}
func Fused1352SiluMulReluName() string { return "silu_mul_relu" }
