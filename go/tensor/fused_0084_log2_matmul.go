package tensor

import "fmt"

func Fused0084Log2Matmul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_matmul: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log2()
	return r, nil
}
func Fused0084Log2MatmulName() string { return "log2_matmul" }
