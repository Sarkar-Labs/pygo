package tensor

import "fmt"

func Fused0800SinAddLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_add_log2: nil operand")
	}
	r := x.Add(y)
	r = r.Sin()
	r = r.Log2()
	return r, nil
}
func Fused0800SinAddLog2Name() string { return "sin_add_log2" }
