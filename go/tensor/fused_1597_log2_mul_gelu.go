package tensor

import "fmt"

func Fused1597Log2MulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul_gelu: nil operand")
	}
	r := x.Mul(y)
	r = r.Log2()
	r = r.GELU()
	return r, nil
}
func Fused1597Log2MulGeluName() string { return "log2_mul_gelu" }
