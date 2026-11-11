package tensor

import "fmt"

func Fused0082Log2Mul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul: nil operand")
	}
	r := x.Mul(y)
	r = r.Log2()
	return r, nil
}
func Fused0082Log2MulName() string { return "log2_mul" }
