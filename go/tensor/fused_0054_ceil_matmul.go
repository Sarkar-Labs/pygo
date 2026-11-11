package tensor

import "fmt"

func Fused0054CeilMatmul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_matmul: nil operand")
	}
	r := x.MatMul(y)
	r = r.Ceil()
	return r, nil
}
func Fused0054CeilMatmulName() string { return "ceil_matmul" }
