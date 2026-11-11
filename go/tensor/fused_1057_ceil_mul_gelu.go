package tensor

import "fmt"

func Fused1057CeilMulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_mul_gelu: nil operand")
	}
	r := x.Mul(y)
	r = r.Ceil()
	r = r.GELU()
	return r, nil
}
func Fused1057CeilMulGeluName() string { return "ceil_mul_gelu" }
