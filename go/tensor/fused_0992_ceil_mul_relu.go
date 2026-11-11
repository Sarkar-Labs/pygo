package tensor

import "fmt"

func Fused0992CeilMulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_mul_relu: nil operand")
	}
	r := x.Mul(y)
	r = r.Ceil()
	r = r.ReLU()
	return r, nil
}
func Fused0992CeilMulReluName() string { return "ceil_mul_relu" }
