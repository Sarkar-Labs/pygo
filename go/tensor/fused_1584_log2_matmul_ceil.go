package tensor

import "fmt"

func Fused1584Log2MatmulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_matmul_ceil: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log2()
	r = r.Ceil()
	return r, nil
}
func Fused1584Log2MatmulCeilName() string { return "log2_matmul_ceil" }
